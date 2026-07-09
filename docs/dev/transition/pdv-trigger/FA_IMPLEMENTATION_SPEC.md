# f_A implementation spec — L1 (fixture screen + edge map) and L2 (production wiring + live matrix), executor-grade

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

**Status (2026-07-06):** spec written; nothing below is implemented yet. This is the
execution-grade expansion of `SOURCE_TERM_DESIGN.md §6` ladder steps **L1** and **L2**, written so
a future session can execute mechanically: every edit site, gate command, pass bar, expected
number, and failure branch is pinned. The design rationale lives in `SOURCE_TERM_DESIGN.md`
(§1–§3) and is NOT repeated here — read that first; the screen result it rests on is
`FINDINGS.md §15`. **Maintainer direction (2026-07-06): a back-reacting in-ODE factor is the
goal; the post-hoc `f_mix` stays production only until f_A earns its way in through this ladder.**

---

## 0. What gets built (one paragraph)

A gated scalar `cooling_boost_fA` (default 1.0 = byte-identical) that multiplies the net
radiative source `dudt` inside the bubble-structure ODE **only in the interface band
(T < 10^5.5 K)** and, consistently, the resolved interface loss integrals L₂ (conduction zone)
and L₃ (intermediate zone) — never the CIE interior L₁, never the conductivity, never the
Eq-44 IC or the Eq-33 dMdt seed. The knob is the production twin of the already-G1-validated
screen injection in `data/make_fA_source_boost.py` — when in doubt about semantics, that harness
is the reference implementation.

## Part A — L1: fixture screen + condensation-edge map (offline, NO production edit, ~1 session)

Cheap extension of the existing screen; run before any production edit. Deliverable: one new
builder run, one CSV + PNG, one FINDINGS subsection (§15a).

**A1. Extend coverage to the 2 captured stiff fixtures.** `make_fA_source_boost.py` currently
covers the 6 cleanroom configs. Add the stiff 5e9 (≈`fail_repro`) + mild-cluster fixtures via the
FM1 loader, copying the exact pattern in `make_kmix_selfconsistent.py` (the
`importlib.util.spec_from_file_location("_fm1", ...make_fm1_rootcheck.py)` block and its
`_fm1._STATES` / `_fm1._load(...)` loop). Gate: G1 identity only (fixtures have no logged
per-row Lloss → no G2), same as the κ_mix harness did. Expected: the 5e9 fixture is the
collapse regime — if it fails at f_A>1 like it did for κ_mix, that is a *recorded edge*, not a
failure of the design (the heavy cloud rides PR #715 anyway; `HIMASS_HANDOFF_PLAN.md`).

**A2. Condensation-edge map.** Per config, per sampled row: bisect f_A upward (grid
{16, 24, 32, 48, 64} then bisect) until the solved `bubble_dMdt ≤ 0` or the solve fails; record
`(config, t_now, f_A_edge, dMdt_at_edge, theta_at_edge)`. Prediction to test (from
`KAPPA_FREEZE_MECHANISM.md §3`): the edge should sit near local θ ≈ 1 (the McKee–Cowie
reversal IS cooling balance) — if edges appear at θ ≪ 1 the "gradual approach" claim of
FINDINGS §15 P4 needs revision. Artifacts: `data/fA_edge_map.csv` + `fA_edge_map.png`
(f_A_edge vs config × epoch, colored by θ_at_edge). ⛔ CONTAMINATION grade: same as §15 —
structural screen, replayed states, no live-fire quotes.

## Part B — L2 production wiring (the two edit sites + registry)

### B1. Registry param (`trinity/_input/registry.py`, insert after `cooling_boost_kappa`, currently :353)

Copy the sibling pattern exactly (`ParamSpec`, `category='input_solver'`, `unit=None`,
`exclude_from_snapshot=True`, `run_const=True`, **no resolver**):

```python
ParamSpec(name='cooling_boost_fA', default='1.0', info='Interface source-term boost f_A (SOURCE_TERM_DESIGN.md): multiplies the net radiative dudt inside the bubble-structure ODE and the resolved L2+L3 loss integrals, ONLY in the interface band T < 10^5.5 K (the non-CIE regime). The 1-D projection of fractal-interface mixing (Lancaster) on the SOURCE side: cooling rises THROUGH the structure and evaporation dMdt FALLS (the El-Badry Eq 47 coupling; contrast cooling_boost_kappa, which raises it). Independent of cooling_boost_mode; default 1.0 = byte-identical. Do not combine with mode=multiplier (double-boost) except deliberately.', category='input_solver', unit=None, exclude_from_snapshot=True, run_const=True),
```

Mirror the same text into `trinity/_input/default.param` next to the `cooling_boost_kappa`
INFO block (currently :293–294). Optional (nice-to-have): a load-time WARNING when
`cooling_boost_fA != 1` AND `cooling_boost_mode == 'multiplier'` (double-boost); a warning, not
an error — sweeps may want it deliberately.

### B2. Edit site 1 — the ODE RHS (`bubble_luminosity.py`, `_get_bubble_ODE`, currently :393–421)

After `dudt = net_coolingcurve.get_dudt(...)` (currently :409) insert:

```python
fA = params['cooling_boost_fA'].value
if fA != 1.0 and T < _T_INTERFACE_BAND:
    dudt = fA * dudt
```

with a module-level constant next to `_T_INIT_BOUNDARY`:

```python
# Interface band top for the f_A source boost = the non-CIE/CIE switch. Duplicated
# as the local _CIEswitch in _bubble_luminosity (same value, kept local there to
# preserve the original diff) -- keep the two in lockstep.
_T_INTERFACE_BAND = 10**5.5
```

The `fA != 1.0` guard makes the default path the **literal production float ops** (the branch is
unreachable) — same construction the screen G1-validated at ≤1.8e-16. Float equality against the
registry literal `1.0` is deliberate and safe.

### B3. Edit site 2 — the loss integrals (`bubble_luminosity.py`, `_bubble_luminosity`)

Immediately before `L_total = L_bubble + L_conduction + L_intermediate` (currently :797):

```python
# f_A scales the interface-band losses consistently with the in-ODE source boost
# (SOURCE_TERM_DESIGN.md §1). L1 (CIE interior, T > 10^5.5 K) is deliberately NOT
# scaled -- there is no mixing interface there.
fA = params['cooling_boost_fA'].value
if fA != 1.0:
    L_conduction = fA * L_conduction
    L_intermediate = fA * L_intermediate
```

Scaling the *components* (not a post-hoc L_eff as the screen did) makes every downstream consumer
consistent automatically: `bubble_LTotal` → the β–δ residual's Lcool, the dataclass fields
`bubble_L2Conduction`/`bubble_L3Intermediate`, and the dictionary/logging. Note |∫f·g| = f·|∫g|
for constant f, so this is exactly the screen's L_eff semantics. Both region-3 sub-branches
(non-CIE and the vestigial CIE mask — the whole L₃ span is 1e4→~3e4 K, so the CIE mask is
normally empty) are covered by scaling the summed `L_intermediate`. `Tavg`, `bubble_mass`,
`T_rgoal` are NOT scaled (the back-reaction already lives in the solved profile).

### B4. What must NOT change (the traps — each was a real failure elsewhere)

1. **Do not touch `net_coolingcurve.get_dudt`.** It has exactly one production caller today (the
   RHS, verified 2026-07-06) but is imported by offline harnesses; the band cut is bubble-model
   logic and belongs in the RHS.
2. **Do not scale L₁** — that is the multiplier's unphysical part we are correcting.
3. **Do not put f_A anywhere near κ**: not in the Eq-44 IC (:377–388), not in the Eq-33 seed
   (:297), not in the RHS conduction prefactor (:413). The whole design rests on the stiff
   operator being untouched (`SOURCE_TERM_DESIGN.md §2`).
4. **Do not introduce a third 10^5.5 literal** — use `_T_INTERFACE_BAND`; leave the local
   `_CIEswitch` assignment untouched (minimal diff) but add the lockstep comment.
5. **Snapshot/metadata collateral (expected, budget for it):** the kappa landing had to touch
   `test_dR2min_magic_number.py` (`_scalar_params`), `test_metadata.py`, `test_mu_audit_drift.py`
   string-pins. Expect the same three; fix by extending the pinned lists, never by weakening the
   tests.

### B5. Tests (new `test/test_fA_source_boost.py` + collateral)

1. **Registry default**: `cooling_boost_fA` present, default 1.0, `run_const`.
2. **Band-limiting unit test**: call `_get_bubble_ODE(r, [v, T, dTdr], params, Pb)` on a synthetic
   state twice (fA=1 vs fA=2 via a params stub): (a) T = 1e6 (> band): identical `[dvdr, dTdr,
   dTdrr]`; (b) T = 1e5 (< band): `dvdr` identical, `dTdrr` differs. (The boost enters only
   through dudt in the dTdrr bracket.)
3. **Component scaling**: with `get_dudt` monkeypatched to a constant, `L_conduction` and
   `L_intermediate` at fA=2 equal 2× their fA=1 values while `L_bubble` is unchanged (or, if a
   full `_bubble_luminosity` call is too heavy for a unit test, assert the block's algebra on the
   returned dataclass of a frozen-state solve, `pytest.mark.slow`).
4. **Default inertness**: fA=1.0 path takes zero new branches — assert via the guard structure
   (code review) + the byte gate below (the real check).

### B6. Gates (in order; each has a pass bar — do not skip rungs; CLAUDE.md rule 5)

1. **Full pytest** green (baseline today: 618 passed).
2. **Byte-identity at default** (the kappa-landing standard, `KAPPA_EFF_SCOPING.md §6.1`):
   run `param/simple_cluster.param` and the stiff `docs/dev/performance/f1edge_hidens*.param`
   pre-patch vs post-patch at default, **separate processes**, `OMP_NUM_THREADS=1
   OPENBLAS_NUM_THREADS=1` (the §9b/fix-#1 A/A lesson: unpinned BLAS gives ULP wobble), and
   compare `sha256sum` of `dictionary.jsonl`. **Mandatory companion: an A/A control** (same code
   twice) — if A/A differs, judge by value-diff instead and record it.
3. **Per-call screen re-run**: `python docs/dev/transition/pdv-trigger/data/make_fA_source_boost.py`
   must reproduce FINDINGS §15 (G1 6/6, G2 6/6, P1–P4 6/6) — the harness now runs against a
   production knob that exists; optionally simplify the harness to set the param instead of
   monkeypatching (keep G1 semantics).
4. **Live smoke (local)**: `simple_cluster` with `cooling_boost_fA 8`, `stop_t 0.1` — runs, no
   crashes, `bubble_dMdt` visibly below the fA=1 companion at matched segments (grep
   `freeze-watch` DEBUG lines), θ visibly above. No θ quotes from this (stop_t rule).

### B7. The live matrix — theta5s (HPC, the calibration measurement)

Mirror the theta5k recipe end-to-end (REPRODUCE.md #33 row is the template):

- **Params builder** `runs/make_theta5s_params.py`, modeled on `runs/make_theta5n_params.py` /
  `runs/make_theta5_params.py`: 9 standard configs (the 8 + `normal_n1e3`) ×
  fA ∈ {1, 2, 4, 6, 8, 12, 16, 24}, `stop_t 5`, `cooling_boost_mode none`.
- **Run**: `sbatch runs/run_theta5s.sbatch` (copy `run_theta5k.sbatch`; keeps the
  `OMP_NUM_THREADS=1` pin), sync, then `runs/harvest_theta_max.py` (θ = `bubble_Lloss/Lmech_total`
  from `dictionary.jsonl`, accepted segments only — NEVER a call-level observer, Retraction R6).
- **Analysis builder** `data/make_theta5s_analysis.py`, modeled on `make_theta5k_analysis.py`:
  fire map, θ_max-vs-fA rise, fates. Plus TWO f_A-specific reads:
  1. **The collapse law, source edition.** Fit `f_fire = A·(0.95/θ₀)^p` as for the multiplier
     (FINDINGS §10). **Registered prediction (2026-07-06, from the screen's θ ∝ f_A^~0.30):
     p_source ≈ 1/0.30 ≈ 3.3**, vs the multiplier's 1.82. A measured p far from ~3 means the
     screen's exponent does not survive live coupling — revise SOURCE_TERM_DESIGN §3 reading (i).
  2. **The fidelity measurement** (impossible for f_mix by construction): harvest `bubble_dMdt(t)`
     per arm, overlay suppression ratio vs the fA=1 arm at matched t, compare the trend against
     El-Badry Eq. 47 `ṁ ∝ (1−θ)^{37/35}/θ^{2/7}` (`ELBADRY_REFERENCE.md §5`). Deliverable:
     `data/theta5s_dmdt_suppression.csv` + figure.
- **Expected window (screen-grade prior, not a pass bar):** whole-band fire somewhere in
  fA ≈ [8, 24]; dense arms may CONDENSE-first (the theta5k race) — that is a fate, not a bug.

### B8. Decision tree after theta5s (pre-commit to these branches; don't relitigate)

| outcome | verdict | action |
|---|---|---|
| a single fA fires the whole 9-config band, proper fates | **f_A is the production-candidate successor** | maintainer ruling on default flip; paper narrative = physical knob with f_mix as its frozen-structure limit; f_mix stays default until ruled |
| no whole-band fA, but θ_max monotone + law fits with rms ≲ multiplier's 0.064 dex | calibrated knob, band incomplete (like kappa but honest fates) | keep f_mix production; f_A becomes the paper's fidelity/appendix knob; record per-config windows |
| dense arms condense before firing across the grid | the McKee–Cowie edge IS the dense transition | acceptable physics (2026-07-02 ruling precedent: fate-based handoff = a transition); document the fA-band + condense-handoff split |
| freezes / no-root grind (θ frozen, exit 0) | P3 falsified live — file it | full stop; instrument with freeze-watch (`KAPPA_FREEZE_MECHANISM.md §5`); do NOT tune around it |
| law exponent p ≪ 3 or ≫ 4 | screen exponent didn't survive coupling | re-measure the response on live arms; update SOURCE_TERM_DESIGN §3(i) and the fA≈8–16↔f_mix≈4 mapping |

### B9. Sibling/bookkeeping checklist for the landing commit(s)

Registry+code+tests commit: update `SOURCE_TERM_DESIGN.md` status line, `FINDINGS.md` (new §),
`PLAN.md` ledger, `INDEX.md` §2/§3, `REPRODUCE.md` (new numbered rows: wiring gate, theta5s),
`CONTAMINATION.md` (register theta5s artifacts + grade), `default.param`, and regenerate
`MANIFEST.md` **in a follow-up commit after** the artifacts land (the manifest reads git history;
see the 2026-07-06 de-dup lesson: after any merge, check `rows == unique rows`).

## Part C — deferred rungs (pointers only; do not start from this doc)

- **L3** (the (★) generalized front IC, saturation cap, condensation branch):
  `SOURCE_TERM_DESIGN.md §5` — a separate workstream with its own rule-5 ladder.
- **L4** (state-coupled f_A): **the L2 wiring IS the L4 plumbing** — the upgrade is replacing the
  scalar read `params['cooling_boost_fA'].value` at the SAME two sites with
  f_A(R2, Pb, λδv, T_pk) from El-Badry's L_int closed form (`ELBADRY_REFERENCE.md §7`). No new
  sites, no new gates class — rerun B6/B7.

## Reproduce / templates

```bash
# L1 (offline, local):
python docs/dev/transition/pdv-trigger/data/make_fA_source_boost.py            # the base screen
# after A1/A2 land, the same builder with EDGE_MAP=1 (implementer's choice of env knob)

# B6.2 byte gate (per side; then diff the shas; plus an A/A control):
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run.py param/simple_cluster.param
sha256sum outputs/<run>/dictionary.jsonl

# B7 (HPC): sbatch runs/run_theta5s.sbatch && runs/sync + harvest_theta_max.py + make_theta5s_analysis.py
```

Templates to copy, not reinvent: `make_kmix_selfconsistent.py` (fixture loader, G1 discipline),
`make_theta5n_params.py` + `run_theta5k.sbatch` + `harvest_theta_max.py` +
`make_theta5k_analysis.py` (the 📏 protocol), `KAPPA_EFF_SCOPING.md §6.1` (byte-gate wording).
