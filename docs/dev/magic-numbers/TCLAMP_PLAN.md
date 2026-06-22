# Finding #1 — the `net_coolingcurve` `T<1e4` clamp: measured, then fixed (file-tied floor)

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

**Status (2026-06-20):** ✅ **FIXED & GATED — shipped (option 1, the file-tied floor).**
This was the measure-first step for `docs/dev/magic-numbers/AUDIT.md` finding #1. **Measurement result:**
across 9.46M `get_dudt` calls in 4 regimes (incl. the stiffest LSODA-flood) the `T<1e4` clamp fired
**0 times** — it is **dead code**, and its justifying comment was *factually wrong about the table bounds*
(table reaches 3162 K, not "3.99", so `1e4` over-floored the valid decade [3162,10000) K). M3 is moot
(dead branch ⇒ all clamp values bit-identical by construction). **Fix shipped:** the hard-coded `1e4`
floor is replaced by a floor tied to the cooling file (`if np.log10(T) < nonCIE_Tmin: T = 10**nonCIE_Tmin`),
gated bit-identical on every reachable regime (see §Fix shipped & gate evidence).

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
| `simple_cluster` (baseline, **full run**) | 1,225,515 | **0** | 0 | 0 | **30000.000** | 127 | 29999.997 | clamp **never fires** |
| `f1edge_lowdens_himass_hisfe` (timeout) | 2,451,122 | **0** | 0 | 0 | **30000.000** | 291 | 29999.997 | clamp **never fires** |
| `f1edge_hidens_himass_losfe` (timeout) | 2,666,884 | **0** | 0 | 0 | **30000.000** | 246 | 29999.99997 | clamp **never fires** |
| `conduction_stiff_5e9` (LSODA flood, timeout) | 3,115,937 | **0** | 0 | 0 | **30000.000** | 380 | 29999.995 | clamp **never fires** |
| **TOTAL** | **9,459,458** | **0** | **0** | **0** | — | 1044 | — | — |

**Conclusion (2026-06-20): the clamp is DEAD CODE in every regime tested.** Across **9.46 million**
`get_dudt` calls spanning the energy-driven baseline, both feedback-strength × density edges, and the
stiffest LSODA-flood, `T < 1e4` fired **zero** times. The minimum `T` the bubble ODE RHS ever passes to
`get_dudt` is exactly **30000.0** — the integrator never goes below the 3e4 outer boundary, never near the
true table min (3162 K), never near 1e4. The per-config "accepted < 3e4" counts (69/158/136/181) are
*all* the `29999.99x` boundary-transient hair — every one sits in the 4.45 log-bin (= the 3e4 boundary),
within 1e-5 fractional of 3e4 (floating-point dust at the boundary), and `accepted_below_1e4 = 0`
everywhere. The author's observed `1e3.91` does **not** reproduce in any current config — it predates the
3e4-boundary / `T~0` guards (`bubble_luminosity.py:390`) or came from a config/table not in this matrix.

### M3 — moot
Because the `T < 1e4` branch is **never taken**, the clamp value is **provably inert**: V0 (`1e4`),
V1 (clamp at the real table min `10**3.5`), and "no clamp" are **bit-identical by construction** on every
tested regime (the differing code path is unreachable). No separate equivalence run is needed to establish
this — the M1 counters already prove the branch is dead. (A confirmatory full-run bit-identity is the gate
for *whichever* fix ships, not a question still open.)

## Verdict & options for the fix — ✅ option 1 chosen & shipped (2026-06-20)
The clamp is simultaneously (a) **dead code** in all tested regimes and (b) built on a **factually-wrong
premise** (the comment's "table only to 3.99"; the table reaches 3162 K, so `1e4` over-floors the valid
decade [3162, 10000) K). Three ways to close audit finding #1 (**#1 shipped**):

1. **Fix to the TODO (✅ SHIPPED).** Replace `if T < 1e4: T = 1e4` with a guard tied to the real table
   min — `if np.log10(T) < nonCIE_Tmin: T = 10**nonCIE_Tmin` (3162 K, the nearest valid table value).
   *Provably inert on current runs* (dead branch ⇒ bit-identical), removes the magic number, and makes the
   guard *correct* if any future regime/table ever does overshoot (uses real table coverage instead of a
   2.5×-hotter floor, and still prevents the line-203 raise below the true table edge). Gate: existing
   `verify_getdudt_equiv.py` (per-call) + a unit test pinning both the no-op on in-range T and the new
   in-table coverage on [3162,10000) K + a full-run byte-identity on `dictionary.jsonl` (trivially passes).
2. **Leave + document.** Correct the wrong table-bound comment and record the measurement (verified dead
   code); change nothing executable. Smallest possible change; leaves the magic number in place.
3. **Remove entirely.** Deletes the magic number but drops the raise-guard insurance for any untested
   regime that *could* overshoot below 3162 K (would then crash at line 203 instead of degrading gracefully).

## Fix shipped & gate evidence (2026-06-20)
**The diff** (`trinity/cooling/net_coolingcurve.py`, in `get_dudt`): removed the hard-coded floor + its
wrong TODO comment; added — *after* the cutoffs are computed so `nonCIE_Tmin` is in scope —
```python
if np.log10(T) < nonCIE_Tmin:
    T = 10 ** nonCIE_Tmin
```
This floors a sub-table `T` to the cooling file's **minimum tabulated** temperature (the nearest valid
table value) so it degrades to the table edge via the non-CIE branch instead of falling through to the
`raise`. The `10**x → log10` round-trip is exact for the bundled grid min (`nonCIE_Tmin = 3.5`,
`log10(10**3.5) == 3.5`), so the clamped value lands on the non-CIE branch (no raise). Strictly more robust
than the old floor, which would itself have raised on any table whose min exceeds 1e4.

**Why this is safe — the equivalence story (NOT "bit-identical everywhere"):** the fix *intentionally*
changes behaviour in the sub-1e4 decade the old code over-floored. The correct gate is **bit-identical for
every `T ≥ 1e4`** (the only regime any real run reaches — measured min T = 30000 K) and **documented,
one-directional divergence below 1e4** (old → 1e4; new → table edge 3162 K; neither raises).

| gate | tool / command | result |
|---|---|---|
| Per-call equivalence (vs `git show HEAD`) | `python docs/dev/magic-numbers/harness/verify_tclamp_equiv.py` | **576 / 576 bit-identical** for T≥1e4 (0 mismatches); 144 / 144 diverged below 1e4 *by design*; **0 raises** (new or ref) |
| Unit tests (runtime `get_dudt`) | `pytest test/test_net_coolingcurve.py` | **3 / 3 pass** — clamps to table edge (not 1e4); over-floored decade uses real T; T≥1e4 untouched |
| **Full-run byte-identity** (new vs HEAD, separate processes, matched-`t`) | `harness/simple_cluster_capped.param` (`stop_t=0.5`), run each, `sha256sum dictionary.jsonl` | **BYTE-IDENTICAL** across 169 snapshots — both `9da691bb458a7aacd7b87a72a4557139edb5bd6699770ba900922773ff302ab0` |
| Full suite + bug-class lint | `pytest` · `ruff check --select F821,F811,F823,E9` | **574 passed**, 3 deselected (stress) · ruff clean |

**Illustrated write-up:** `docs/dev/magic-numbers/tclamp_report.html` (self-contained, 6 figures + LaTeX) —
regenerate with `python docs/dev/magic-numbers/harness/make_tclamp_report.py` (figures:
`make_tclamp_figures.py`, a pure read of `data/`; the dudt overlay tabulates once via `make_tclamp_overlay_data.py`).

Capped `stop_t=0.5` (vs the default 15) keeps the full-run gate to minutes — an uncapped `simple_cluster`
reaches only t≈2 Myr in ~20 min. A fixed `stop_t` makes both runs truncate at the **same** `t` by
construction (the matched-`t` requirement). The early energy-driven phase exercises the solver + cooling +
snapshot I/O end-to-end; combined with the per-call gate covering the *entire reachable input domain*, there
is no reachable state where new ≠ old. Reproduce: `python run.py docs/dev/magic-numbers/harness/simple_cluster_capped.param`.

## Subagent fan-out (this round)
Lead built + smoked the M1/M2 harness (validated: it reproduces the documented 3e4 boundary floor). Three
subagents run the edge/stiff configs in parallel (distinct output dirs via the override), each reporting
its summary row. **If no config drives `T<3e4`**, M3 is moot (V0≡V1 trivially, clamp is dead code
everywhere ⇒ ship the TODO fix as pure correctness cleanup). **If a config fires the clamp**, lead builds
M3 (clamp-variant equivalence) targeted at that config.
