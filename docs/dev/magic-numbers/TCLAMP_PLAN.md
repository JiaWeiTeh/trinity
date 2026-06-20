# Finding #1 — the `net_coolingcurve` `T<1e4` clamp: measure before fixing

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
> and full runs cost minutes-to-hours, so any diagnostic worth keeping must be
> saved as a committed artifact under `docs/dev/` — never left in `/tmp`, the
> local-only `scratch/`, or an untracked `outputs/`. A future visit must be able
> to reproduce or compare against the numbers **without re-running**; record the
> exact config + command that produced each artifact.

**Status (2026-06-20):** 🔵 **PLAN — facts verified, measurement not yet run.** This is the
measure-first step for `docs/dev/magic-numbers/AUDIT.md` finding #1. No production code will change
until the gate below is cleared. **Headline correction from verification (see §Verified facts):** the
clamp's own justifying comment is *factually wrong about the table bounds*, which upgrades this from a
"harmless guard" to a "guard on a false premise that over-floors a valid decade of temperature."

## The code under audit
`trinity/cooling/net_coolingcurve.py:122-123`:
```python
if T < 1e4:
    T = 1e4
```
with the comment (`:114-120`) — *"TODO: this in the future has to depend on the file … follow the
minimum temperature of the cooling file … the temperature seem to run at some very low value
(~1e3.91) and the lowest available value of the cooling file … is only until 3.99 … Not sure why
though, as the temperature should be around 1e7, not 1e4."*

## Verified facts (measured / source-read 2026-06-20 — NOT assumed)
1. **Single caller.** `get_dudt` is called from exactly one production site: `bubble_luminosity.py:402`,
   inside `_get_bubble_ODE` (the bubble-structure ODE RHS). It is **not** called from the shell. (grep:
   only other hits are comments, the equivalence harness, and a drift test.)
2. **The clamp prevents an Exception, not a NaN.** Below the non-CIE table's `Tmin`, `get_dudt` falls
   through every branch to `else: raise Exception` (`:202-203`). Clamping `T` up to `1e4` keeps
   `log10(T)=4.0` inside the non-CIE branch so it never raises.
3. **The table bounds — the comment is WRONG.** Measured from the real default tables loaded by
   `param/simple_cluster.param` (script: reuse `verify_getdudt_equiv._setup_params`):
   - non-CIE grid: **log10 T ∈ [3.5, 5.5]** = **3162 K … 316 228 K**, 21 pts.
   - CIE grid: log10 T ∈ [4.0, 10.0].
   So the true raise-boundary is **3162 K (log 3.5)**, not 1e4. `T = 1e3.91 = 8128 K` is **inside** the
   table (3.5 ≤ 3.91 ≤ 5.5) and would **not** raise. ⇒ the clamp over-floors the whole valid decade
   **[3162 K, 10000 K)** up to 1e4, substituting the `Λ(1e4)` rate for correct in-table rates. The
   author's premise ("table only to 3.99") does not hold for the current tables.
4. **It can steer the solve.** The returned `dudt` enters `dTdrr` (`bubble_luminosity.py:408`, the
   `- dudt/Pb` term), so changing the rate at a trial point changes the ODE RHS there → can change which
   steps `solve_ivp` accepts. So even a "trial-only" clamp is **not automatically** inert to results.
5. **Geometry of the excursion.** The bubble boundary is `_T_INIT_BOUNDARY = 3e4 K` (log 4.477) and the
   bubble integrates *inward* to ~1e8 K, so a valid profile has `T ≥ 3e4` everywhere. `T < 1e4` is a deep
   sub-boundary value ⇒ **hypothesis:** these are `solve_ivp` trial/rejected-step overshoots, not the
   accepted profile. To be confirmed by M2, not assumed.

## Hypotheses
- **H1 (expected):** every `T<1e4` event is an integrator *trial-step* overshoot; the **accepted** bubble
  profile never dips below ~3e4 K; and changing the clamp threshold to the real table min is bit-identical
  (or within solver tolerance) on final results. ⇒ the fix is a **safe correctness cleanup**: replace the
  magic `1e4` with a guard tied to `nonCIE_Tmin` (exactly what the TODO asks), handling `T<Tmin` by the
  table edge rather than a 2.5×-higher temperature.
- **H2:** low-T lands in **accepted** profiles, **or** the clamp *value* changes final results. ⇒ deeper
  issue (the bubble genuinely cooling below boundary, or the over-floor biasing the solution); needs the
  full gate and possibly an upstream fix.

## Experiments (all read-only / non-invasive; production untouched)
Configs (cheap → stiff): `param/simple_cluster.param` (baseline), `docs/dev/performance/f1edge_lowdens_*.param`,
`f1edge_hidens_*.param`, `docs/dev/performance/conduction_stiff_5e9_sfe001.param` (the LSODA-flood regime,
most likely to overshoot). Each bounded by a wall-clock `timeout` (full runs don't bound on `stop_t`).

- **M1 — frequency & location.** Non-invasive wrapper on `net_coolingcurve.get_dudt` records every call's
  `(t_now, current_phase, T, ndens, phi)` and calls through unchanged. Per config: fraction of calls with
  `T<1e4` and with `T<3162` (the true raise-boundary), min T reached, and the phase/regime where they cluster.
- **M2 — accepted vs trial.** Capture each bubble solve's **final sampled T-profile** `min` (the array used
  for luminosity, e.g. the `Bubble structure: r=[...], T=[min,max]` debug). If accepted `T_min ≈ 3e4` always,
  the low-T events are trial-only ⇒ H1 geometry holds.
- **M3 — impact / equivalence gate (decisive).** Run each config in **separate processes** with the clamp
  replaced by variants, compare final `dictionary.jsonl` / `dMdt`/`L` at **matched `t`**:
  - V0 = production (`T<1e4 → 1e4`).
  - V1 = guard at the **true table min** (`T < 10**nonCIE_Tmin → 10**nonCIE_Tmin`), i.e. the TODO fix.
  - V2 = (diagnostic) clamp far lower (e.g. `1e2`) to see if the integrator ever *depends* on the floor.
  Bit-identical V0≡V1 ⇒ clamp value is inert to results ⇒ cleanup is free (just correctness + a comment).
  V0≠V1 ⇒ the over-floor matters ⇒ quantify and gate.

## Decision tree (what each outcome implies)
- **H1 + V0≡V1 (most likely):** ship the TODO fix (tie guard to `nonCIE_Tmin`, handle `<Tmin` at the table
  edge). It is a correctness improvement (uses real table coverage [3162,10000) K) that is provably inert on
  current results. Pin with a `test_*.py` (the no-op + the new in-table coverage). Smallest diff.
- **H1 + V0≠V1:** the over-floor changes the solve via trial-step steering. Quantify the delta on the stiff
  edge; pick the variant that is *correct* (in-table rate) and gate it as a result-changing fix.
- **H2 (low-T in accepted profile):** stop — the bubble is cooling below its boundary; that is the real bug
  the comment half-saw. Escalate to a boundary/stiffness fix, not a cooling-table tweak.

## Results

### Harness
`harness/tclamp_instrument.py` — non-invasive wrappers on `get_dudt` (M1) and `_solve_bubble_structure`
(M2); accumulates counters + a log10(T) histogram and dumps the < 3e4 K excursions row-by-row. SIGTERM/
atexit flush ⇒ a `timeout`-killed stiff run still persists partial data. Optional 4th arg overrides
`path2output` so parallel runs don't collide. Artifacts: `data/<tag>_summary.json`, `data/<tag>_lowT.csv`.

### M1/M2 by config
| config | get_dudt calls | T<1e4 | T<3162 | T<3e4 | min T (any RHS) | accepted solves | accepted min T | verdict |
|---|---|---|---|---|---|---|---|---|
| `simple_cluster` (baseline, **full run**) | 1,225,515 | **0** | 0 | 0 | **30000.000** | 127 | 29999.997 | clamp **never fires**; floor = boundary |
| `f1edge_lowdens_himass_hisfe` | _pending subagent_ | | | | | | | |
| `f1edge_hidens_himass_losfe` | _pending subagent_ | | | | | | | |
| `conduction_stiff_5e9` (LSODA flood) | _pending subagent_ | | | | | | | |

**Baseline reading (2026-06-20):** in 1.2M RHS evals the minimum `T` ever passed to `get_dudt` is exactly
`30000.0` — the integrator never goes below the 3e4 boundary. The 69 "accepted < 3e4" are the
`29999.997` boundary-transient hair (log 4.477, within 1e-5 of 3e4; the documented `min_T` noise), not
real sub-boundary excursions; every accepted profile floors at the boundary (`accepted_minT_hist` = all
127 in the 4.45 bin). ⇒ on the energy-driven baseline the clamp is **provably inert** — it is dead code
for this regime. The author's observed `1e3.91` must come from a stiffer regime (or older path); the edge
configs test that.

## Subagent fan-out (this round)
Lead built + smoked the M1/M2 harness (validated: it reproduces the documented 3e4 boundary floor). Three
subagents run the edge/stiff configs in parallel (distinct output dirs via the override), each reporting
its summary row. **If no config drives `T<3e4`**, M3 is moot (V0≡V1 trivially, clamp is dead code
everywhere ⇒ ship the TODO fix as pure correctness cleanup). **If a config fires the clamp**, lead builds
M3 (clamp-variant equivalence) targeted at that config.
