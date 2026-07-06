# H3 experiment: is "Eb going non-positive" the SOLE failure mode? (Eb-floor diagnostic)

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
> a committed artifact (a CSV/table under `docs/dev/data/`, or a force-added
> harness/figure in the relevant `docs/dev/<workstream>/` folder) — never left in
> `/tmp` or an untracked `outputs/`. A future visit must be able to reproduce or
> compare against the numbers **without re-running**; record the exact config +
> command that produced each artifact.
>
> 🔗 **Cross-check the sibling docs — keep the workstream self-consistent.** This file is one of
> several living docs for its workstream (its `PLAN.md`, `FINDINGS.md`, `runs/README.md`, `NOTE_PATCHES.md`,
> and any other notes in the same folder). They drift out of sync *with each other* as fast as they drift
> from the code. Any agent or person editing one MUST, as part of the visit, circle back through the
> siblings and reconcile: if a number, status, claim, or line reference here contradicts a sibling — or a
> sibling has gone stale — fix it (or flag it, dated) so no two docs in the workstream disagree. Never
> update one in isolation.

**Date:** 2026-06-22. Branch `fix/transition-trigger-problem-pt4`. DIAGNOSTIC experiment;
production untouched (monkeypatch-only). Line refs verified against current source on this date.

---

## 0. The energy-injection caveat (READ FIRST)

The `EBFLOOR` variant **FLOORS the bubble thermal energy `Eb` to a small positive value so it
can never collapse through zero.** This **INJECTS ENERGY and VIOLATES energy conservation.** It
is a **DIAGNOSTIC experiment, NOT a production fix candidate.** Its sole purpose: isolate whether
"Eb going non-positive" is the **SOLE** failure mode for the massive-cloud collapse regime — i.e.
whether the rest of the physics stays intact (shell keeps expanding, v2 physical, solver healthy)
once `Eb` is forced `> 0`. A floor that "fixes" the run does so by manufacturing energy; it cannot
ship.

## 1. Hypothesis (H3, the maintainer's)

Massive clouds (`mCloud=5e9`) drain `Eb` via PdV expansion work (`4πR²·Pb·v2 > Lmech`), so `Eb`
collapses through zero and the run hits the shipped `ENERGY_COLLAPSED` clean stop
(`docs/dev/failed-large-clouds/PLAN.md`, treat as unverified; verified §3 below).
*(Update 2026-07-01: that dead-stop is exactly what H3/H4 argued should instead be a momentum handoff —
now shipped for a finite `Eb<=0` in phase 1b, which ROUTES to momentum. This experiment predates the fix.
See `docs/dev/transition/pdv-trigger/HIMASS_HANDOFF_PLAN.md`.)* H3 asks: **if we
artificially floor `Eb > 0`, does the bubble keep expanding and the run proceed cleanly — does the
failing cloud then behave like a healthy one** (the "healthy vs failing" diagnostic,
storyline3 §1.4 / `failed-large-clouds/figures/fig2`: healthy ⇒ `Eb` grows, `PdV/Lmech<1`;
failing ⇒ `Eb` collapses)?

## 2. Variant design, injection point, and what it bypasses

Harness: `h3_variants.py` (monkeypatch), `h3_run_variant.py` (one sim/process driver),
`h3_run_matrix.sh` (matrix), `h3_salvage_timeout.py` (recover partial jsonl on a grind/timeout),
`h3_analyze.py` (tabulate). Copied from `docs/dev/failed-large-clouds/harness/{variants,run_variant}.py`;
V0/V1/V2/V3 carried over verbatim, `EBFLOOR` is the new piece. Originals NOT edited.

The shipped collapse picture, verified on current source:
- `Eb` is consumed by `get_bubbleParams.bubble_E2P(Eb,R2,R1)` (→Pb,
  `get_bubbleParams.py:198-238`) and `solve_R1(R2,Eb,…)` (→R1, `:413-446`). As `Eb→0` the wind
  shock `R1→R2`, `(R2³−R1³)→0`, and the divide blows up; the shipped geometry guard floors
  `shell_volume` at `1e-13·r2³` (`:229-235`) and `solve_R1` returns `0.0` for `R2<=0` (`:433`),
  so the divide stays finite.
- The shipped **`ENERGY_COLLAPSED` early-stops read the ODE STATE `Eb`** extracted from
  `solution.y[:,-1]`: **`run_energy_phase.py:340`** (`if not np.isfinite(Eb) or Eb <= 0`) and
  **`run_energy_implicit_phase.py:1007`** (same). Code **51 = `ENERGY_COLLAPSED`**, in the
  **inspection band 50-59** (`simulation_end.py:90`, `is_inspection_required` 50-59) — a clean
  termination, not a crash.

`EBFLOOR` therefore needs TWO coupled pieces (both monkeypatch module attributes):

- **(A) Keep the DRIVE positive.** Clamp `Eb = max(Eb, floor)` at entry to `gbp.bubble_E2P`
  (→ `Pb>0`) and `gbp.solve_R1` (→ `R1` well-defined). Reached everywhere via the module attribute
  (energy RHS `energy_phase_ODEs.py:223,226`, the bubble solve, diagnostics). Includes the V3
  geometry guards internally so the divide can't blow up.
- **(B) Keep the STATE positive — the new piece.** A **reflecting floor on `dEb/dt`**: whenever the
  state `Eb <= floor`, the energy-derivative component returned to `solve_ivp` is clamped `>= 0`, so
  the integrated state can only hold flat or grow at the floor and never crosses below it. Wraps the
  two RHS functions each phase's `solve_ivp` reads:
  - phase 1a: `energy_phase_ODEs.get_ODE_Edot_pure` (module-attr call at `run_energy_phase.py:272`);
  - phase 1b: `run_energy_implicit_phase.get_ODE_implicit_pure` (bare-name call at
    `run_energy_implicit_phase.py:929`; its 3rd return is `Ed_from_beta`, the betadelta cooling
    derivative actually integrated — clamping that controls the 1b state).

  Verified bindings (`h3_variants.py` patches both module namespaces): 1a uses the patchable module
  attribute `energy_phase_ODEs.get_ODE_Edot_pure`; 1b's `get_ODE_implicit_pure` is a bare name in
  `run_energy_implicit_phase`'s namespace.

**WHAT (B) BYPASSES:** by construction the state never reaches `<= 0`, so the shipped
`ENERGY_COLLAPSED` clean stops (`run_energy_phase.py:340`, `run_energy_implicit_phase.py:1007`)
**never fire**. That is the experiment — and exactly why this is not a fix (it manufactures energy).

**Floor choice.** `floor = 1e-3` [au = Msun·pc²/Myr²], a fixed small positive value ~13 orders of
magnitude below the collapse regime's initial `E0 ≈ 6.4e9` au (so the floor only bites once the
bubble has genuinely drained to near-zero). Sensitivity: the activation telemetry (`act_state`,
`min_Eb_seen`) shows whether the floor ever engaged; see §5.

## 3. Results — full all-configs matrix

Run: `bash docs/dev/transition/pt4/h3_run_matrix.sh` (OMP_NUM_THREADS=1, one sim/process, each cell
`timeout`-bounded with a small `--stop_t`). Per-cell rows in `h3_eval.csv`; trajectories in
`traj/h3_traj_<cfg>_<variant>.csv`; driver log `h3_run_matrix.log`. Outcomes: `crashed` /
`completed` (clean `end_reason`) / `timeout` (grind — partial jsonl salvaged by `h3_salvage_timeout.py`).

Full table (`h3_analyze.py` → `h3_summary.txt`; one row per (config, variant); `rmin` =
min cooling-balance ratio `(Lgain−Lloss)/Lgain`; trigger = `rmin < 0.05`):

| config | class | var | outcome | reached | end_code | final_t | final_R2 | final_Eb | floor_act | trig | rmin |
|---|---|---|---|---|---|---|---|---|---|---|---|
| simple_cluster | stall | V0 | completed | 1b | 1 | 0.005 | 0.371 | 1.15e6 | False | F | 0.666 |
| simple_cluster | stall | EBFLOOR | completed | 1b | 1 | 0.005 | 0.371 | 1.15e6 | True* | F | 0.666 |
| large_diffuse_lowsfe | stall | V0 | completed | 1b | 1 | 0.005 | 1.882 | 4.27e6 | False | F | 0.968 |
| large_diffuse_lowsfe | stall | EBFLOOR | completed | 1b | 1 | 0.005 | 1.882 | 4.27e6 | True* | F | 0.968 |
| small_dense_highsfe | stall | V0 | completed | 1b | 1 | 0.005 | 0.2615 | 2.23e5 | False | F | 0.496 |
| small_dense_highsfe | stall | EBFLOOR | completed | 1b | 1 | 0.005 | 0.2615 | 2.23e5 | True* | F | 0.496 |
| midrange_pl0 | stall | V0 | completed | 1b | 1 | 0.005 | 0.744 | 4.14e6 | False | F | 0.852 |
| midrange_pl0 | stall | EBFLOOR | completed | 1b | 1 | 0.005 | 0.744 | 4.14e6 | True* | F | 0.852 |
| pl2_steep | stall | V0 | completed | 1b | 1 | 0.005 | 0.455 | 3.75e6 | False | F | 0.705 |
| pl2_steep | stall | EBFLOOR | completed | 1b | 1 | 0.005 | 0.455 | 3.75e6 | True* | F | 0.705 |
| be_sphere | stall | V0 | completed | 1b | 1 | 0.005 | 0.647 | 2.06e6 | False | F | 0.839 |
| be_sphere | stall | EBFLOOR | completed | 1b | 1 | 0.005 | 0.647 | 2.06e6 | True* | F | 0.839 |
| **fail_repro** | collapse | V0 | **completed** | 1b | **51** | 0.00341 | 9.73 | **−9.14e8** | False | F | 0.997 |
| **fail_repro** | collapse | **EBFLOOR** | **TIMEOUT(grind)** | 1a | — | 0.00291 | 8.61 | 4.75e8 | killed | F | — |
| **fail_helix** | collapse | V0 | completed | 1a | **51** | 0.00255 | 7.03 | 0.00509 | False | F | — |
| **fail_helix** | collapse | **EBFLOOR** | completed | 1a | **51** | 0.00255 | 7.03 | 0.00509 | True* | F | — |
| mass_5e8 | collapse | V0 | completed | 1b | 1 | 0.01 | 11.40 | 2.93e9 | False | F | 0.984 |
| mass_5e8 | collapse | EBFLOOR | completed | 1b | 1 | 0.01 | 11.40 | 2.93e9 | False | F | 0.984 |
| mass_1e9 | collapse | V0 | TIMEOUT(slow) | 1b | — | 0.00683 | 10.78 | 2.67e9 | killed | F | 0.989 |
| mass_1e9 | collapse | EBFLOOR | TIMEOUT(slow) | 1b | — | 0.00733 | 11.23 | 3.01e9 | False | F | 0.989 |
| small_1e5 | healthy | V0 | completed | 1b | 1 | 0.005 | 1.163 | 4.37e5 | False | F | 0.956 |
| small_1e5 | healthy | EBFLOOR | completed | 1b | 1 | 0.005 | 1.163 | 4.37e5 | True* | F | 0.956 |
| small_1e6 | healthy | V0 | completed | 1b | 1 | 0.005 | 1.882 | 4.27e6 | False | F | 0.968 |
| small_1e6 | healthy | EBFLOOR | completed | 1b | 1 | 0.005 | 1.882 | 4.27e6 | True* | F | 0.968 |
| small_1e7 | healthy | V0 | completed | 1b | 1 | 0.005 | 3.104 | 3.99e7 | False | F | 0.978 |
| small_1e7 | healthy | EBFLOOR | completed | 1b | 1 | 0.005 | 3.104 | 3.99e7 | False | F | 0.978 |

`* floor_act=True` on a no-op config = transient sub-floor probes during `solve_ivp` step
rejection; the **accepted trajectory is identical to V0** (§5). `killed` = process SIGTERM'd at
the `timeout`; the partial jsonl was salvaged (`h3_salvage_timeout.py`).

**Three things jump out:**
1. **`rmin` is 0.50–0.99 everywhere it is defined — never within 100× of the 0.05 trigger.** The
   cooling-balance transition trigger **never fires** under EBFLOOR (`trig=F` on every row), exactly
   as H1/H2 predicted: flooring `Eb` adds *drive*, not *cooling*. (On configs that stop in 1a —
   `fail_helix`, both `fail_repro`/`fail_helix` collapse — `rmin` is undefined because no implicit
   `Lloss`/`Lgain` snapshot was reached.)
2. **Only `mCloud=5e9` (`fail_repro`, `fail_helix`) actually collapses** at these early times.
   `mass_5e8` completes cleanly with `Eb=2.9e9>0`; `mass_1e9` times out **healthy-and-slow** (12–13
   implicit snapshots, `Eb≈2.7–3.0e9>0`, R2 growing to ~11 pc — *not* a grind; see §4/§5). So the
   collapse regime is narrow and mass-gated, consistent with `failed-large-clouds`.
3. **EBFLOOR does NOT rescue either real-collapse config**: `fail_repro` → implicit-phase **grind**
   (timeout, 0 implicit rows); `fail_helix` → **identical** `ENERGY_COLLAPSED` via a *second* guard
   the floor does not touch (§4).

## 4. Trajectory evidence — the collapse configs

Of the four "collapse" configs, **only the two `mCloud=5e9` runs actually collapse** at these early
times; `mass_5e8`/`mass_1e9` stay healthy (`Eb>0`) in the window (§3) — so the real collapse regime is
just `fail_repro` and `fail_helix`. They fail by **two different shipped paths**, and the Eb floor
defeats neither into a clean expansion:

**`fail_repro` (mCloud=5e9, PISM=1e4) — EBFLOOR turns a clean stop into an implicit-phase GRIND.**
- V0: 52 energy-phase snapshots, then **one** implicit step drives `Eb` 4.75e8 → **−9.14e8**, and the
  shipped `Eb<=0` guard (`run_energy_implicit_phase.py:1007`) stops cleanly (`ENERGY_COLLAPSED` 51) at
  t=0.00341, R2=9.73 pc. Runtime 80s.
- EBFLOOR: the energy phase is **bit-identical to V0** (52 matched-t rows, `max|ΔR2|=0`,
  `max rel|ΔEb|=0`) — the floor never bit on the accepted energy trajectory (`Eb≥4.75e8 ≫ 1e-3`).
  At the energy→implicit handoff (t=0.00291, R2=8.61, v2=2344) the implicit `solve_ivp` **grinds**:
  it hammers the RHS with **1,206,110 drive-clamps + 603,056 state-clamps** (sub-floor `Eb` probes
  during step rejection) and **never advances a single implicit step** (0 implicit snapshots) →
  killed by the 300s `timeout`. The reflecting floor (`Ed_from_beta` clamped `≥0` when `Eb≤floor`)
  injects energy on every sub-floor probe, fighting the betadelta solver's attempt to drive `Eb`
  down, so the step never converges. **R2 does NOT keep growing — it freezes at 8.61 pc; the run
  makes zero time progress for ~5 min.** This is exactly the V3 "solver grinds forever" failure mode
  noted in `failed-large-clouds/PLAN.md`, now with the floor actively engaged.

**`fail_helix` (mCloud=5e9, PISM=0) — EBFLOOR is byte-identical to V0; a SECOND collapse path my
floor does not touch.** Both V0 and EBFLOOR stop at t identical, `ENERGY_COLLAPSED` (51), reached
**1a** (never reaches the implicit phase), `final_Eb=0.005085287954767937` (identical to the digit),
55s. The stop fires via a **different guard**: `run_energy_phase.py:169-183` wraps the bubble-
structure solve `bubble_luminosity.get_bubbleproperties_pure(params)` in a try/except — as `Eb→0`
the **bubble STRUCTURE integration itself degenerates** (cooling table out of bounds, `solve_R1`
cannot bracket) and raises, mapped to `ENERGY_COLLAPSED` with reason *"bubble solve degenerate as
Eb -> 0"*. The Eb floor (which only clamps `Eb` inside `bubble_E2P`/`solve_R1` and clamps `dEb/dt`)
**does not prevent the structure solve from breaking down**, so the same clean stop fires regardless.
floor_act counters (drive=530, state=265) are transient probes; the trajectory is identical to V0.

**Cooling-balance trigger under EBFLOOR:** NOT fired on any collapse config (`trigger_fired=False`,
as expected per H1/H2 — flooring `Eb` adds drive, not cooling). On `fail_repro` the salvaged
trajectory has no `Lloss`/`Lgain` rows at all (it never reached an implicit snapshot).

## 5. No-op confirmation — stall + healthy configs

Where `Eb` never collapses, `EBFLOOR` is designed to be identical to V0 (the `if Eb < FLOOR`
branch is never taken on the accepted trajectory). Confirmed by matched-snapshot diff
(`h3_traj_sample.py --noop <cfg>`, comparing every row of `traj/h3_traj_<cfg>_{V0,EBFLOOR}.csv`):

| config | V0 outcome | EBFLOOR outcome | floor_act (drive,state) | max\|ΔR2\| [pc] | max rel\|ΔEb\| | no-op? |
|---|---|---|---|---|---|---|
| simple_cluster | STOPPING_TIME(1) | STOPPING_TIME(1) | True (6,3) | 0 | 0 | **bit-identical** |
| large_diffuse_lowsfe | STOPPING_TIME(1) | STOPPING_TIME(1) | True (4,2) | 0 | 0 | **bit-identical** |
| midrange_pl0 | STOPPING_TIME(1) | STOPPING_TIME(1) | True (4,2) | 0 | 0 | **bit-identical** |
| pl2_steep | STOPPING_TIME(1) | STOPPING_TIME(1) | True (6,3) | 0 | 0 | **bit-identical** |
| small_dense_highsfe | STOPPING_TIME(1) | STOPPING_TIME(1) | True (10,5) | 1.5e-11 | 5.9e-8 | track-identical (fp) |
| be_sphere | STOPPING_TIME(1) | STOPPING_TIME(1) | True (4,2) | 0 | 0 | **bit-identical** |
| small_1e5 | STOPPING_TIME(1) | STOPPING_TIME(1) | True (4,2) | 2.0e-10 | 6.8e-9 | track-identical (fp) |
| small_1e6 | STOPPING_TIME(1) | STOPPING_TIME(1) | True (4,2) | 0 | 0 | **bit-identical** |
| small_1e7 | STOPPING_TIME(1) | STOPPING_TIME(1) | False (0,0) | 0 | 0 | **bit-identical** |
| mass_5e8 | STOPPING_TIME(1) | STOPPING_TIME(1) | False (0,0) | 0 | 0 | **bit-identical** |
| mass_1e9 | TIMEOUT(slow,healthy) | TIMEOUT(slow,healthy) | False (0,0) | 0 (first 90 rows) | 0 | **bit-identical** (both slow) |
| **fail_helix** (collapse) | ENERGY_COLLAPSED(51) | ENERGY_COLLAPSED(51) | True (530,265) | 0 | 0 | **bit-identical** (same 2nd-path stop) |
| **fail_repro** (collapse) | ENERGY_COLLAPSED(51) | TIMEOUT/grind | killed | — | — | **DIFFERS** (energy phase bit-identical for all 52 rows; diverges only at the implicit handoff) |

(Full per-snapshot diffs: `h3_noop_diffs.txt`, generated by `h3_traj_sample.py --noop`.)

Across **11 of 13** configs EBFLOOR is **bit-identical** to V0 on every accepted snapshot
(`max|ΔR2|=0`, `max rel|ΔEb|=0`); `small_dense_highsfe` and `small_1e5` differ only at the **~1e-8 /
1e-9** level (a single accepted `solve_ivp` step whose internal probe dipped below `FLOOR`, nudging
it) — a no-op to ~8 significant figures. **`mass_1e9` is bit-identical** to V0 for its first 90
snapshots (both are slow healthy runs that hit the wall clock, not grinds). The only true divergence
is **`fail_repro`**, and even there the **entire energy phase (52 snapshots) is bit-identical** —
EBFLOOR and V0 part ways *only* at the energy→implicit handoff, where V0 takes one step to `Eb<0`
and stops cleanly while EBFLOOR grinds.

**Key nuance — `floor_activated=True` does NOT mean the trajectory changed.** The `act_drive` /
`act_state` counters tally *every RHS evaluation* in which `Eb < FLOOR`, **including `solve_ivp`'s
rejected trial steps** during adaptive error control. On these healthy/stall runs the accepted
trajectory is **bit-identical to V0** (`max|ΔR2| = 0`, `max rel|ΔEb| = 0`) on 4 of 5 configs;
`small_dense_highsfe` differs only at the **1e-8 level** (a single accepted step whose internal
probe dipped below `FLOOR`), i.e. a no-op to ~8 significant figures. So the few counter hits are
transient adaptive-stepping probes, not energy injection that perturbs the physics. **The Eb floor
is a genuine no-op wherever Eb never collapses — exactly the "test on all configs" payoff.**

## 6. Verdict

**Is "Eb going non-positive" the SOLE failure mode — does everything else hold once `Eb>0`? NO.**
Forcing `Eb>0` does **not** make either failing cloud behave like a healthy one. Two independent
failure modes survive the floor:

1. **`fail_repro` (mCloud=5e9, PISM=1e4): the implicit betadelta solver GRINDS once the clean
   `Eb<=0` stop is removed.** EBFLOOR's energy phase is bit-identical to V0 (Eb never reaches the
   floor; 52/52 snapshots match), but at the energy→implicit handoff the first implicit `solve_ivp`
   segment **never converges** — it hammers the RHS with ~1.2M drive- and ~0.6M state-clamps (probing
   `Eb` as deep as −2.6e7), makes **zero time progress for 5 minutes**, and is killed by the
   `timeout`. **R2 does not keep growing; it freezes at 8.61 pc.** This is precisely the V3 "solver
   grinds forever" mode (`failed-large-clouds/PLAN.md`). The shipped `ENERGY_COLLAPSED` stop is not
   an arbitrary cutoff being papered over — it is the only thing that *lets the run end cleanly*
   instead of grinding, because the energy-driven interior is genuinely degenerate as `Eb→0`.

2. **`fail_helix` (mCloud=5e9, PISM=0): a SECOND, independent collapse guard the floor cannot
   bypass.** Both V0 and EBFLOOR stop **byte-identically** via `run_energy_phase.py:169-183` — the
   1a-phase try/except around `bubble_luminosity.get_bubbleproperties_pure`, which fires
   `ENERGY_COLLAPSED` (reason *"bubble solve degenerate as Eb -> 0"*) when the **bubble-structure
   integration itself breaks down** (cooling table out of bounds, `solve_R1` can't bracket). My floor
   clamps `Eb` in `bubble_E2P`/`solve_R1` and clamps `dEb/dt`, but it does **not** keep the structure
   solve from degenerating, so the same stop fires regardless. Eb-collapse-through-zero is therefore
   only *one* of (at least) two ways the energy-driven model breaks in this regime.

**Corollaries (measured, not assumed):**
- **The cooling-balance transition trigger never fires under EBFLOOR** (`rmin` 0.50–0.99 ≫ 0.05 on
  every config). Flooring `Eb` adds *drive*, not *cooling* — it cannot manufacture a cooling-balance
  event, fully consistent with H1 (bubble too hot to radiate) and H2 (under-cooling is intrinsic to
  the energy-conserving interior). H3 changes nothing here.
- **The floor IS a clean no-op wherever `Eb` does not collapse** — bit-identical to V0 on 11/13
  configs (≤1e-8 on the other two). So the experiment is well-controlled: EBFLOOR only does something
  in the genuine collapse regime, and there it makes things *worse* (grind), not better.
- **The collapse regime is narrow and mass-gated:** only `mCloud=5e9` collapses early; `5e8`/`1e9`
  stay healthy (`Eb>0`) in the window. This sharpens `failed-large-clouds`: the PdV-drain collapse is
  a high-cluster-mass phenomenon, not generic to "big clouds."

**Bottom line.** The shipped `ENERGY_COLLAPSED` termination is **diagnosing a real physics breakdown,
not hiding a tractable continuation.** Propping `Eb>0` does not yield a healthy, expanding bubble; it
either (a) leaves a *different* degeneracy guard to fire (fail_helix) or (b) replaces a clean stop
with a non-convergent grind (fail_repro). A genuine fix for the massive-cloud regime must be a real
**momentum-driven continuation** (or added cooling/leakage physics), not an energy floor — which, as
flagged up front, injects energy and cannot ship anyway. **H3 refuted: Eb-collapse is not the sole
failure mode; the energy-driven interior is degenerate as `Eb→0` along more than one axis.**

## 7. Reproduce / artifacts (all committed under `docs/dev/transition/pt4/`)

Reproduce the whole matrix (each cell one sim/process, OMP_NUM_THREADS=1, timeout-bounded):
```
bash docs/dev/transition/pt4/h3_run_matrix.sh        # -> h3_eval.csv, traj/*.csv, h3_run_matrix.log
python docs/dev/transition/pt4/h3_analyze.py          # -> the §3 table + §5 no-op check (h3_summary.txt)
python docs/dev/transition/pt4/h3_traj_sample.py --noop <cfg>   # per-snapshot V0-vs-EBFLOOR diff
```
One cell directly:
`OMP_NUM_THREADS=1 timeout 300 python docs/dev/transition/pt4/h3_run_variant.py --variant EBFLOOR
--param docs/dev/failed-large-clouds/harness/params/fail_repro.param --stop_t 0.01 --floor 1e-3
--csv /tmp/x.csv --traj /tmp/x_traj.csv --out /tmp/h3/fail_repro_EBFLOOR`

- `h3_variants.py` — EBFLOOR monkeypatch (Eb clamp in `bubble_E2P`/`solve_R1` delegating to the
  ORIGINAL helpers + reflecting `dEb/dt≥0` state floor in both phases' RHS) + V0/V1/V2/V3.
- `h3_run_variant.py` — one-sim/process driver; reads the run's `dictionary.jsonl` for trajectory,
  cooling-balance trigger, floor-activation telemetry.
- `h3_run_matrix.sh` — full (config, variant) matrix with per-cell timeout + stop_t (stall/healthy
  stop_t=0.005, collapse stop_t=0.01; timeouts 300s).
- `h3_salvage_timeout.py` — recovers partial jsonl into an eval row + trajectory on a grind/timeout.
- `h3_analyze.py` — tabulates `h3_eval.csv` + the matched-state no-op check → `h3_summary.txt`.
- `h3_traj_sample.py` — sampled trajectory printer + per-snapshot `--noop <cfg>` diff.
- `h3_eval.csv` — one row per (config, variant): outcome, end_code, final state, floor activation
  counters, `min_Eb_seen`, trigger, `ratio_min`, runtime. (`mass_1e9`/`fail_repro` have a runner row
  + a salvage row from the timeout; the salvage row is authoritative.)
- `traj/h3_traj_<cfg>_<variant>.csv` — per-snapshot (t, phase, R2, v2, Eb, Pb, R1, T0, Lgain, Lloss,
  ratio, rCloud). 26 files (13 configs × {V0, EBFLOOR}).
- `h3_summary.txt`, `h3_noop_diffs.txt` — committed renderings of §3 and the §5 diffs.
- `h3_run_matrix.log` — driver log (exact per-cell command lines + outcomes + salvage records).
