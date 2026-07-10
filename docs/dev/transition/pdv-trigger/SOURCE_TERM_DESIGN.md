# f_A — the source-term cooling boost: design, evidence, and THE execution workflow (single source of truth)

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

**Status (2026-07-06, consolidated):** THE single plan doc for the f_A workstream — maintainer
directive: one stream of workflow, no parallel plan docs. The former `FA_IMPLEMENTATION_SPEC.md`
was folded into §3 here and **deleted** the same day; if you find a reference to it, it means
this doc. Design (§1) and screen evidence (§2) are settled; the workflow (§3) is the execution
stream for the next sessions: Phase 0 ✅ done, **Phase 1 ✅ done (2026-07-06, `FINDINGS.md §15a`)**,
Phases 2–6 ⬜ open. Two review agents audited this plan on 2026-07-06 (config-coverage audit;
literature-benchmark extraction) — their findings are integrated throughout and marked "(audit)" /
"(lit)". **Phase 1 headline: the condensation-edge prediction (edges near θ≈1) was FALSIFIED in the
SAFE direction — no dMdt≤0 edge exists for f_A even at 512 (16× the physical range); the source knob
structurally cannot reach the f_κ condensation crash. All 9 configs now have offline coverage.**

---

## 0. Orientation (read first)

**What f_A is.** A gated scalar `cooling_boost_fA` (default 1.0 = byte-identical) multiplying the
net radiative source `dudt` **inside** the bubble-structure ODE and, consistently, the resolved
interface loss integrals L₂ (conduction zone) + L₃ (intermediate zone) — **only in the interface
band T < 10^5.5 K**. Never the CIE interior L₁, never the conductivity, never the Eq-44 IC or the
Eq-33 dMdt seed, never the leakage. It is the 1-D projection of fractal-interface turbulent
mixing (Lancaster) placed on the **source** side, where a scalar stays physical, tunable, and
solver-safe. The maintainer's goal (2026-07-06): a back-reacting in-ODE factor to replace the
post-hoc `f_mix` as the production knob, calibrated against published bubble simulations.

**f_A vs the knobs that exist (the one-table orientation):**

| | `f_mix` (production) | κ_mix (shelved) | f_κ (probe) | **f_A (this plan)** |
|---|---|---|---|---|
| acts on | L_cool *after* solve | diffusivity in ODE | Spitzer C in ODE+ICs | **source `dudt` in ODE, band-limited** |
| structure feels it | no | yes | yes | **yes** |
| evaporation ṁ | untouched | (port unstable) | **rises** (wrong sign) | **falls** (El-Badry sign) |
| dial? | linear | born-saturated | dial, crash-prone | **dial, θ ∝ f_A^~0.3** |
| CIE interior L₁ | boosted (unphysical) | — | over-conducted | untouched |

**Why κ_mix can't be rescued with small λδv** (maintainer FAQ 2026-07-06): its knee is at
λδv ≈ 0.01 — 300× below the physical λδv ≈ 3 — because κ_S = C·T^{5/2} collapses in the cool
layer and R = κ_mix/κ_S is already 10⁵–10⁸ at any plausible magnitude
(`KMIX_SELFCONSISTENT.md §2a`). Operating on the rising part (λδv ~ 0.001–0.01) makes the
parameter meaningless (a fudge with units) and razor-steep (whole θ range in ~half a dex). The
saturation is structural — any T-flat floor swamps a T^{5/2} law at low T. The enhancement cannot
live in the diffusion coefficient; that is why it lives in the source.

**Session kickoff template (paste to start any executor session on this workstream):**

> Read `docs/dev/transition/pdv-trigger/SOURCE_TERM_DESIGN.md` in full — it is the single source
> of truth for the f_A workstream. Then execute **the next open phase of §3 only** (one phase per
> session; the Status line and INDEX say which). Before coding, honor the ⚠️/🔄 banners:
> re-verify the phase's cited line references against current source and update the doc where it
> drifted. Branch: `git fetch origin`; if the workstream branch's PR has merged, restart it from
> `origin/main` (same name), else continue on it. Hard rules: no production edits outside the
> phase's specified sites; θ only from `dictionary.jsonl` accepted rows; every artifact committed
> with its builder + exact command; on completion write the FINDINGS entry, add the REPRODUCE
> row, reconcile INDEX/PLAN, regenerate MANIFEST.md in a follow-up commit, run full pytest + the
> docs-conventions test; commits carry no AI attribution. **If a result falsifies a registered
> prediction, STOP and write the finding up — do not tune around it.** End by updating this doc's
> Status line and phase checkboxes.

Maintainer-only inputs (the executor must ask, not guess): HPC sbatch submission + sync (Phases
4–5), the L21b Table-1 values (Phase 5 pre-step), and the rulings in Phase 6.

## 1. Design rationale (settled 2026-07-06 — do not relitigate; §2 is the evidence)

The knob 2×2 (κ vs source × scalar vs state-coupled): f_κ (κ×scalar) is tunable but couples
evaporation the **wrong way** (measured ṁ ×1.08–1.17 up at f_κ=2, `KAPPA_EFF_SCOPING.md §6a`;
El-Badry's resolved hydro wants ÷3–30 *down*, Eq. 47: ṁ ∝ (1−θ)^{37/35}/θ^{2/7}) and walks the
solver onto the condensation domain edge (`KAPPA_FREEZE_MECHANISM.md`); κ_mix (κ×state-coupled)
is faithful but born-saturated and boundary-divergent; `theta_target` double-counts PdV
(demoted). The empty corner — **source-side, in-ODE** — is f_A.

Four literature anchors for the source side: (1) **El-Badry+19 result vi** — "mixing, not
conduction, sets the cooling; Spitzer mainly sets interior T/evaporation" — the roles are
separable, f_κ conflated them. (2) **Lancaster+21a/b** — the interface is a fractal turbulent
mixing layer radiating over enlarged area; in 1D that is an area/emissivity factor on the
interface-band n²Λ, not a transport change. (3) **Weaver+77 §V** — even the classical front
radiates ~40% of the conductive flux; the evaporate-vs-radiate *split* is the physical dial and
f_A moves it directly. (4) **Eq. 47's sign** — a source boost suppresses evaporation
automatically (radiated flux no longer evaporates); a conductivity boost cannot.

**The honest counter-argument and its resolution.** Gupta+18 (MNRAS 473, 1537) is a published
precedent for f_κ (κ = 6e-7·f_T·T^{5/2}, T_c ∝ f_T^{−2/7}, verified), and a fractal surface
wrinkles the conductive contact area too — on paper the 1D projection is ambiguous. The
discriminator is empirical: the two projections back-react on ṁ with **opposite signs**, and the
resolved-hydro ground truth says down. Scaling algebra ([D]-grade, re-verify exponents before
hard-coding): T_c ∝ f_κ^{−2/7}, ṁ ∝ f_κ^{2/7}, L_cool ∝ f_κ^{4/7–5/7}; an output multiplier
f_mix *implies* f_κ = f_mix^{7/5} while dropping the ṁ ×f_mix^{2/5} and T_c ×f_mix^{−2/5}
side-effects it owes — order-unity state errors in exactly the quantities conduction physics
controls, and the dropped ṁ factor has the wrong sign anyway. f_A drops neither.

**Why f_A cannot reproduce the two structural failures:** no operator takeover (source term is
linear in the RHS → no κ_mix-style saturation), and no stiff-machinery perturbation (dR2, anchor
gradient, hot-interior conduction untouched → no f_κ-style crash chain; its only route to the
condensation edge is genuinely radiating the front's budget away — which fix #1's
`no_physical_root_handoff` (`KAPPA_FREEZE_MECHANISM.md §7`) already routes to momentum).

f_κ itself stays available as the genuine *conduction* knob — suppression f_κ ∈ [~1e-3, 1]
(Markevitch & Vikhlinin 2007 draped interfaces ≤1e-2; Chandran & Cowley 1998 tangled fields;
Narayan & Medvedev 2001 ~0.2), enhancement only as a bounded area factor ≲30 under a
Cowie–McKee saturation cap with a Dalton–Balbus *smooth* limiter (never a hard min — the κ_mix
hard-max lesson). That, plus the generalized front IC, lives in the deferred track (§4).

## 2. Evidence to date (Phase 0, measured 2026-07-06 — `FINDINGS.md §15`)

Offline screen `data/make_fA_source_boost.py` (6 cleanroom configs × ~10 replayed C0 rows ×
f_A ∈ {1,2,4,8,16}; monkeypatch, no production edit; G1 identity 6/6 (≤1.8e-16 = 1-ULP equivalence, NOT literal byte-identity: the screen
assembles L1+f_A·(L2+L3) vs production's (L1+L2)+L3 association), G2 replay 6/6 ≤3.1e-7). All four registered predictions passed 6/6:

| config | n | θ_max @ f_A=1→2→4→8→16 | ṁ @ 2→16 | solves |
|---|---:|---|---|---|
| large_diffuse_lowsfe | 1e2 | 0.52→0.56→0.65→0.84→**1.21** | 0.98→**0.88** | 50/50 |
| be_sphere | 1e4 | 0.50→0.53→0.60→0.73→**0.99** | 0.98→**0.88** | 50/50 |
| midrange_pl0 | 1e4 | 0.50→0.52→0.57→0.68→0.89 | 0.98→**0.89** | 50/50 |
| pl2_steep | 1e5 | 0.50→0.52→0.56→0.66→0.85 | 0.98→**0.90** | 50/50 |
| simple_cluster | 1e5 | 0.61→0.64→0.71→0.86→**1.18** | 0.98→**0.89** | 50/50 |
| small_dense_highsfe | 1e6 | 0.55→0.58→0.65→0.80→**1.10** | 0.97→**0.85** | 50/50 |

P1 dial (smooth, monotone, no saturation, no dense ceiling) ✓ · P2 El-Badry sign (ṁ falls
monotonically — the first TRINITY knob to do so) ✓ · P3 stability (300/300 incl. the early
high-Pb epochs that NaN'd κ_mix, at 2× the f_κ crash point) ✓ · P4 no condensation cliff in
range ✓. Response is sub-linear, θ_max ∝ f_A^~0.30 *globally* — but per-config effective
exponents vary (pl2_steep ≈0.19 → the laggards need larger f_A than naive inversion suggests;
this drives the Phase-4 grid). ⛔ CONTAMINATION: replayed frozen states — structural verdicts
only, **no fire threshold quotable**.

**Verified plumbing identities (2026-07-06, against source — the executor relies on these):**
- `bubble_Lloss = bubble_LTotal + bubble_Leak` (`trinity_reader.py:239`), written from
  `effective_Lloss` at `run_energy_implicit_phase.py:930`; harvested θ = bubble_Lloss/Lmech_total.
- **L_leak is deliberately NOT scaled by f_A** — leakage is bulk escape through shell
  perforations, not interface radiation. The referee-facing identity: θ numerator =
  L₁ + f_A·(L₂+L₃) + L_leak. Leak is nonzero exactly at late/fragmenting epochs (diffuse θ peak)
  — report the Lcool/Lleak split in every harvest (the Rogers & Pittard degeneracy, §3 Phase 5).
- **Phase 1a parity is automatic**: `trinity/phase1_energy/run_energy_phase.py:170` solves the
  same bubble structure and `:279` routes its trigger through `effective_Lloss_from_params` — f_A
  acts there with no extra wiring (matters for dense configs firing at the 1a/1b boundary,
  t≈0.003 Myr).
- **Pre-existing latent bug (flagged, NOT ours to fix in this diff):** the trigger fallback at
  `run_energy_implicit_phase.py:1245–1247` re-applies `effective_Lloss_from_params` to the
  already-effective `bubble_Lloss` when `bubble_props is None` — a double-boost for
  mode='multiplier', harmless for 'none'/f_A. Registered in `FINDINGS.md §16`.

## 3. THE WORKFLOW — the single execution stream (phases in order; do not skip)

> Executor notes that apply to every phase: shadow-first (harnesses before production edits);
> separate processes, `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` for any byte/A-A comparison
> (FINDINGS.md §9b ULP lesson); θ only from `dictionary.jsonl` accepted rows via `runs/harvest_theta_max.py`
> (never call-level observers — Retraction R6, `SESSION_HANDOFF_2026-07-01.md §6`); every artifact committed with its builder +
> command; reconcile siblings + regenerate `MANIFEST.md` in a follow-up commit (after any merge,
> check MANIFEST `rows == unique rows` — the 2026-07-06 de-dup lesson); re-verify the pytest baseline (733 selected at the 2026-07-06 post-merge HEAD; PLAN's older
> entries say 617/618 — counts move with main) and all line references at execution time.

### Phase 0 ✅ — offline screen (done; §2 above)

### Phase 1 ✅ — offline completeness: all-9 coverage + the condensation-edge map (done 2026-07-06, `FINDINGS.md §15a`)

> RESULT: all 9 configs + 2 FM1 fixtures now have offline f_A coverage (2 new committed
> trajectories `data/traj_{normal_n1e3,small_1e6}.csv`, both PARTIAL/early-epoch — the §8d cliff
> makes ≥5 Myr infeasible in-container). Both controls stay far below fire (small_1e6 θ_max=0.25
> @ fA=16; stiff-5e9 fixture θ≈0.02). Condensation-edge map: **0/50 states reach dMdt≤0 within
> f_A≤128, and a probe to f_A=512 finds no edge** (θ driven to 6–26, dMdt stays positive). The
> registered θ≈1 edge prediction is FALSIFIED — but in the safe direction: the source knob has no
> reachable condensation edge (it never touches the evaporative eigenvalue), so it cannot hit the
> f_κ crash. Builder `data/make_fA_edge_map.py`. **Consequence for Phase 6:** the "dense arms
> condense-first" row means DRAIN or stay-energy-driven, not an f_A-driven condensation.

Original spec (kept for provenance):

1. **Extend the screen to the missing configs** (audit G3). The §2 screen covered only the 6
   cleanroom configs. Add: (a) the 2 captured FM1 fixtures (stiff 5e9 ≈ fail_repro **with the
   caveat its sfe=0.01 ≠ fail_repro's 0.1** — state it in the output; and mild cluster), copying
   the `importlib` loader block from `make_kmix_selfconsistent.py`; (b) **small_1e6 and
   normal_n1e3**, which have no committed replay trajectories: run each locally at default to
   stop_t ≈ 0.5–1 Myr (both are minutes–tens-of-minutes configs), capture per-segment state
   (the cleanroom C0 CSV column pattern) or 3–5-epoch JSON fixtures, **commit them** under
   `data/`, and extend the loader. These two are the band's boundary definers (route-a control;
   native-fire config) — they must not meet f_A for the first time on HPC.
2. **Condensation-edge map** (per config × epoch): raise f_A on grid {16, 24, 32, 48, 64} then
   bisect until `bubble_dMdt ≤ 0` or solve failure; record (config, t_now, f_A_edge, θ_at_edge).
   **Registered prediction:** edges sit near local θ ≈ 1 (the McKee–Cowie reversal IS cooling
   balance); edges at θ ≪ 1 falsify the "gradual approach" reading of §2/P4.
   **Row selection** (audit G10): the ~10 even rows PLUS force-include the first 3 accepted
   implicit rows (dense race window, t<0.06 Myr) and each config's documented θ-peak epoch
   (diffuse: t≈4.9 Myr) — the same force-include pattern `make_da_replay` uses for blowout.
3. Builder: `data/make_fA_edge_map.py` (may import/extend `make_fA_source_boost.py`; env knobs
   documented in its docstring). Artifacts: `data/fA_edge_map.csv` + `fA_edge_map.png` + the new
   fixtures; FINDINGS §15a; REPRODUCE row. Same ⛔ grade as §2 (replayed states).

### Phase 2 ⬜ — production wiring (two edit sites + registry + tests)

**Registry** (`trinity/_input/registry.py`, insert after `cooling_boost_kappa`, currently :353;
same ParamSpec shape: `category='input_solver'`, `unit=None`, `exclude_from_snapshot=True`,
`run_const=True`, no resolver):

```python
ParamSpec(name='cooling_boost_fA', default='1.0', info='Interface source-term boost f_A (docs/dev/transition/pdv-trigger/SOURCE_TERM_DESIGN.md): multiplies the net radiative dudt inside the bubble-structure ODE and the resolved L2+L3 loss integrals, ONLY in the interface band T < 10^5.5 K (the non-CIE regime). The 1-D projection of fractal-interface mixing (Lancaster) on the SOURCE side: cooling rises THROUGH the structure and evaporation dMdt FALLS (El-Badry Eq 47 coupling; contrast cooling_boost_kappa, which raises it). L_leak is deliberately NOT scaled (leakage is bulk escape, not interface radiation). Requires f_A > 0; values < 1 are untested suppression territory. Default 1.0 = byte-identical. Single-knob use intended: combining with cooling_boost_mode != none or cooling_boost_kappa != 1 warns at load (double-boost / cross-knob).', category='input_solver', unit=None, exclude_from_snapshot=True, run_const=True, validator=_validate_cooling_boost_fA),
```

Define `_validate_cooling_boost_fA(value, params)` next to `_validate_dens_profile`
(`registry.py:108` — validators receive `(value, params)`): raise on f_A ≤ 0, and emit the
required cross-knob WARNING there too (f_A ≠ 1 with `cooling_boost_mode != 'none'` or
`cooling_boost_kappa != 1`/`'auto'`) — one home for both; if load ordering bites (kappa-'auto'
resolves at load), move the warning to the end of the `read_param` load path and note it.

Mirror the text into `default.param` next to the kappa block (currently :293–294). **Required
(audit G6, promoted from optional): a load-time WARNING** when `cooling_boost_fA != 1` and
(`cooling_boost_mode != 'none'` or `cooling_boost_kappa != 1`, including `'auto'`) — and add to
the kappa-`'auto'` registry text: "grid measured at cooling_boost_fA=1" (the 819-run lookup is
silently invalidated by f_A≠1).

**Edit site 1 — ODE RHS** (`bubble_luminosity.py`, `_get_bubble_ODE`, currently :393–421; after
the `dudt = net_coolingcurve.get_dudt(...)` line, currently :409):

```python
fA = params['cooling_boost_fA'].value
if fA != 1.0 and T < _T_INTERFACE_BAND:
    dudt = fA * dudt
```

with a module-level constant next to `_T_INIT_BOUNDARY`:

```python
# Interface band top for the f_A source boost = the non-CIE/CIE switch. THREE places
# must stay in lockstep: this constant, the local _CIEswitch in _bubble_luminosity,
# and the cooling-table-derived nonCIE_Tcutoff in net_coolingcurve._noncie_cutoffs
# (they coincide on the default bundle; a table swap moves the third — see the
# pinning unit test in test_fA_source_boost.py).
_T_INTERFACE_BAND = 10**5.5
```

**Edit site 2 — loss components** (`_bubble_luminosity`, immediately before
`L_total = L_bubble + L_conduction + L_intermediate`, currently :797):

```python
# f_A scales the interface-band losses consistently with the in-ODE source boost.
# L1 (CIE interior) and L_leak are deliberately NOT scaled (no mixing interface
# there / bulk escape). |int f*g| = f*|int g| for constant f, so component scaling
# equals the screen's L_eff = L1 + fA*(L2+L3) while keeping the dataclass,
# residual (Lcool), dictionary logging, and harvest chain consistent automatically.
fA = params['cooling_boost_fA'].value
if fA != 1.0:
    L_conduction = fA * L_conduction
    L_intermediate = fA * L_intermediate
```

The `fA != 1.0` guards make the default path the literal production float ops (screen-validated
construction, G1 ≤1.8e-16). `Tavg`, `bubble_mass`, `T_rgoal` are NOT scaled (the back-reaction
lives in the solved profile).

**Tests** (new `test/test_fA_source_boost.py` + collateral):
1. registry default present, 1.0, run_const; rejects fA ≤ 0.
2. band-limiting unit test: `_get_bubble_ODE` on a synthetic state, fA=1 vs 2: T=1e6 → identical
   triplet; T=1e5 → `dvdr` identical, `dTdrr` differs.
3. component scaling: with `get_dudt` monkeypatched constant, L₂ and L₃ double at fA=2, L₁
   unchanged (frozen-state solve acceptable, `pytest.mark.slow`).
4. **band-edge pin** (audit G9): `_noncie_cutoffs` returns **log10** grid values (verified:
   `(5.5, 3.5)` on the default bundle, compared against `np.log10(T)` at
   `net_coolingcurve.py:138`) — so assert IN LOG SPACE:
   `_noncie_cutoffs(cube)[0] == np.log10(_T_INTERFACE_BAND)`. A future cooling-table swap
   (theta5c) then fails loudly instead of silently splitting the f_A band from the L₂ mask.
5. interaction warning fires (fA=2 + mode='multiplier'), and the double-boosted L_loss is what
   the docs say it is.
6. expected string-pin collateral (verified live): `test_dR2min_magic_number.py:98`
   `_scalar_params` (add the new key), `test_metadata.py`, `test_mu_audit_drift.py`. Fix by
   extending pinned lists, never by weakening tests.

**Traps (each was a real failure elsewhere — do not):** touch `net_coolingcurve.get_dudt` (one
production caller today, but harnesses import it; the band cut is bubble-model logic); scale L₁
or L_leak; put f_A near κ (not the Eq-44 IC :377–388, not the Eq-33 seed :297, not the RHS
conduction prefactor :413); introduce a third 10^5.5 literal.

### Phase 3 ⬜ — gates (in order; pass bars pinned)

1. **Full pytest** green (baseline 733 selected at 2026-07-06 HEAD — re-verify at execution).
2. **Byte-identity at default**: `param/simple_cluster.param` +
   `docs/dev/performance/f1edge_hidens*.param`, pre- vs post-patch, separate processes, pinned
   threads, `sha256sum dictionary.jsonl`, **plus a mandatory A/A control** (same code twice; if
   A/A differs, judge by value-diff and record). Budget note: simple_cluster at default stop_t is ~90 min/run
   in-container; pre + post + the mandatory A/A pair = 4–6 runs (~6–9 h serial). **Coverage justification (audit G7):** f1edge_lowdens is
   omitted because the default path is unreachable-branch-inert and the screen's G1 covers
   diffuse states; if a reviewer objects, add lowdens at matched `stop_t 0.7` (past its ~0.61 Myr
   blowout) to both sides.
3. **Screen re-run**: `python docs/dev/transition/pdv-trigger/data/make_fA_source_boost.py` must
   reproduce §2 (G1/G2 6/6, P1–P4 6/6) against the now-real param (optionally set the param
   instead of monkeypatching; keep G1 semantics).
4. **Live smoke**: simple_cluster, `cooling_boost_fA 8`, `stop_t 0.1`, DEBUG on (freeze-watch
   lines are `logger.debug` — enable via the `log_level DEBUG` param / the §8d runner mechanism,
   NOT plain run.py defaults): runs clean; `bubble_dMdt` visibly below the fA=1 companion at
   matched segments; θ visibly above. No θ quotes (stop_t rule).

### Phase 4 ⬜ — theta5s: the all-9-config live matrix (HPC)

- **Grid** (audit G2 ⊕ lit prediction): f_A ∈ **{1, 2, 4, 6, 8, 12, 16, 24, 32}** × 9 configs =
  81 arms, `stop_t 5`, mode=none, kappa=1 (single-knob by construction — say so in the sbatch
  header). Rationale: lit-side inversion from live θ₀ predicts the whole-band value in
  **[8, 13]**; the screen's laggard endpoints (midrange 0.89, pl2 0.85 at 16, effective exponent
  ≈0.19–0.30) extrapolate their crossings to ~20–32; the grid brackets both. **Bracket rule
  (pre-committed):** if any fireable config is NOFIRE at 32, submit {48, 64} for that config
  before reading the decision tree.
- **Wall-time armor** (audit G4): `--time=6:00:00` minimum (f_A is in-ODE → early-segment cost
  like FINDINGS.md §8d's f_κ, AND Eb-back-reacting → dt shrink like Retraction R4's f_mix
  [`SESSION_HANDOFF_2026-07-01.md §6`]; fA=24–32 exceeds any boost
  run live; diffuse arms are the long pole). Keep `.exit_code`/`.duration` writes. **Mandatory
  post-harvest compliance gate**: every arm shows `t_final ≥ 5` or a physics termination; any
  wall-kill/nonzero-exit arm is re-run longer before ANY θ is quoted (protocol rule 2) — report
  "N/81 compliant" like theta5's 32/32.
- **Builders**: `runs/make_theta5s_params.py` (template: `make_theta5n_params.py`), arm naming
  pinned `<config>__fa<value>` with `.`→`p` (`fa2p5` style); `runs/run_theta5s.sbatch` (copy
  theta5k's, bump time); harvest via `runs/harvest_theta_max.py`; analysis
  `data/make_theta5s_analysis.py` (template: `make_theta5k_analysis.py`).
- **Reads**: (i) fire map + θ_max rise + fates; (ii) **collapse law, source edition**: fit
  f_fire = A·(0.95/θ₀)^p. Registered prediction p_source ≈ 1/0.30 ≈ 3.3 (vs multiplier 1.82),
  with the caveat that per-config exponents vary (0.19–0.30) so the law may fit worse than the
  multiplier's rms 0.064 dex — report the rms either way; (iii) **the fidelity measurement**
  (impossible for f_mix by construction): per arm, interpolate `bubble_dMdt(t)` onto the fA=1
  arm's accepted-row time grid (the compare_live.py pattern), report suppression only where
  overlap ≥ ~20 segments, arms firing before ~0.1 Myr contribute upper limits only; compare the
  *trend* against El-Badry Eq. 47 — a trend check, not a fit (state divergence at matched t is
  conflated in the ratio; say so in the caption). Deliverable
  `data/theta5s_dmdt_suppression.csv` + figure.

**Per-config expected outcomes & acceptance (audit A — "works on ALL configurations" is
per-CLASS, not "all fire"):**

| config | mCloud / nCore / sfe | θ₀ (5 Myr) | expected f_A outcome | PASS criterion |
|---|---|---|---|---|
| simple_cluster | 1e5 / 1e5 / 0.3 | 0.676 | FIRES ~6–12 (condense-race risk at high f_A: its κ condensed at 8+) | fires with proper fate |
| small_dense_highsfe | 1e4 / 1e6 / 0.5 | 0.717 | FIRES or CONDENSE-handoff early (photo-finish, t<0.06 Myr; §14 domain-edge NaN class now rescued) | fires or labeled condense fate |
| pl2_steep | 1e6 / 1e5 / 0.1 | 0.511 | laggard — fires near grid top (~24–32) or DRAIN | fires within grid ∪ bracket rule |
| midrange_pl0 | 1e6 / 1e4 / 0.1 | 0.636 | laggard — fires ~20–24 or DRAIN ceiling | fires within grid ∪ bracket rule |
| be_sphere | 1e6 / 1e4 / 0.05 | 0.529 | FIRES ~16 | fires with proper fate |
| large_diffuse_lowsfe | 1e7 / 1e2 / 0.01 | 0.535 (peak t≈4.9) | FIRES ~8–16; horizon-sensitive near stop_t; runtime long pole | fires (allow t≈5 crossing per theta5b precedent) |
| normal_n1e3 | 1e6 / 1e3 / 0.01 | 1.047 | fires natively at fA=1; watch high-f_A drain/condense race (κ drained at 16) | fires at 1; no pathological high-f_A fate |
| fail_repro (control) | 5e9 / 1e2 / 0.1 | 0.003 | PdV-dominated: rides PR#715 Eb≤0→momentum at every f_A | **unchanged**: same fate, handoff t within ~1 segment of fA=1 |
| small_1e6 (control) | 1e6 / 1e2 / 0.1 | 0.297 | route-a: stays energy-driven (law extrapolates f_fire ≈ 50–70) | does **NOT** fire; healthy to 5 Myr; θ_max < 0.95, monotone in f_A |

Headline question: does a single f_A fire **7/7 fireable** configs (the multiplier's gold
standard was [4, 4.5] at 7/7) while both controls pass *unchanged*? **Never tune f_A to make the
controls fire — that would itself be a bug.**

### Phase 5 ⬜ — literature calibration: matched-config benchmarks (lit agent, 2026-07-06)

The maintainer's criterion: *an f_A value is working well if it produces similar θ at similar
time for a cloud config found in published bubble simulations.* Primary anchor: **Lancaster
et al. 2021b** (ApJ 914, 90 — 3D hydro, winds-only, Θ ≡ Ė_cool/L_w = 0.9–0.99 with
1−Θ ∝ t^{−1/2}, α_p ≈ 1.2–4, suite spanning n̄ ≈ 40–2×10⁵ cm⁻³ — squarely TRINITY's regime).
Secondary: El-Badry+19 closed form (shape/asymptote check, off-regime at n ≤ 10), Lancaster 2025
CEM pair (winds+radiation+B direction-of-bias bracket; also the precedent — semi-analytic vs 3D
RMHD scored at ~25%).

**⚠ Pre-step (blocking):** the mapping below rests on suite details recovered by web search
(M_* = 5000 M⊙; V_w = 3230/1759 km/s; exact M–R pairings) — **verify against the L21b PDF
Table 1 and add to `LANCASTER_REFERENCE.md`** (imprint protocol) before freezing the .params.
**Fallback if PDFs are unreachable in the environment (they were on 2026-07-06 — proxy blocks
non-GitHub):** ask the maintainer to supply Table 1 (values needed: M_*, V_w, the M–R pairings),
or freeze the .params from the search-snippet values above with an explicit [I]-grade provenance
note in `runs/params/bench5/README` — do NOT silently treat them as verified.

**Benchmark configs** (5 bespoke `.param`s, `runs/params/bench5/` — flat profile `densPL`
α=0 ⇒ nCore = n̄ exactly; sfe = 5000/M_cl; check `rCloud_max` plausibility validation passes):

| L21b model (M_cl, R_cl) | n̄_H | t_ff | TRINITY mapping |
|---|---:|---:|---|
| 5e4 M⊙, 20 pc | ≈43 | ≈6.6 Myr | mCloud 5e4, nCore 43, sfe 0.1 |
| 1e5 M⊙, 10 pc | ≈690 | ≈1.7 Myr | mCloud 1e5, nCore 6.9e2, sfe 0.05 |
| 1e5 M⊙, 5 pc | ≈5.5e3 | ≈0.6 Myr | mCloud 1e5, nCore 5.5e3, sfe 0.05 |
| 1e5 M⊙, 2.5 pc | ≈4.4e4 | ≈0.2 Myr | mCloud 1e5, nCore 4.4e4, sfe 0.05 |
| 5e5 M⊙, 2.5 pc | ≈2.2e5 | ≈0.09 Myr | mCloud 5e5, nCore 2.2e5, sfe 0.01 |

Run the bespoke params — the standard band members are only loose stand-ins (mass/density
mismatches up to 200×).

**Protocol.** Per benchmark × f_A ∈ {1, 4, 6, 8, 12, 16} (+ the Phase-4 winner), `stop_t 5`:
two arms — (a) **diagnostic** with the cooling trigger disabled via
`transition_trigger blowout` (a legal token set, `run_energy_implicit_phase.py:249`; blowout
still transitions at R2 > rCloud, which is W's upper cap anyway) so θ(t) logs uncensored through
the energy phase, (b) **production** with the default `cooling_balance` trigger live for fire
time/fate. Comparison window W = [t_first (the first accepted implicit row — TRINITY logs no
explicit shell-formation time), min(3 Myr, t(R2=R_cloud), stop_t)] — the 3 Myr cap keeps
`Lmech_total` ≈ L_wind so the Θ definitions align (SB99 SNe start ~3–4 Myr); if the SB99 reader
exposes the wind-only channel, prefer θ_w = Lloss/L_wind and drop the cap.

**Metrics & pass bands** (compute both statistics — L21b's headline Θ is *cumulative*, TRINITY's
θ is *instantaneous*):
1. Θ_cum(t_end) = ∫_W L_loss dt / ∫_W L_mech dt ∈ **[0.9, 0.99]** — THE primary pass band; a
   benchmark passes iff Θ_cum is in-band (Phase 6 references this criterion, stated only here).
   Bench-1 caveat (t_ff ≈ 6.6 Myr): all matched epochs collapse to the single 3 Myr cap, so
   bench-1 contributes Θ_cum only, not the trajectory statistic.
2. Trajectory, in 1−θ space (absolute-θ tolerances are meaninglessly tight near 1):
   **|log₁₀(1−θ_TRINITY) − log₁₀(1−Θ_L21b)| ≤ 0.5 dex** at matched epochs
   t* ∈ {0.5, 1, 2}·t_ff (capped at 3 Myr); fitted slope d log(1−θ)/d log t ∈ [−1, 0]
   (L21b: −0.5).
3. Optional dynamics cross-check: shell momentum / ṗ_w·t within α_p ∈ [1.2, 4].
4. **Censoring rule** (applies ONLY if a diagnostic arm is missing — with one, trajectories
   always come from it): a production-arm fire before t* is *Lancaster-consistent*, not failure;
   score the censored epoch against the Θ_cum band (a fired arm pins θ ≥ 0.95, inside [0.9,
   0.99]) — do NOT apply the dex-trajectory tolerance to a censored point (0.95 vs Θ = 0.99 is
   0.7 dex in 1−θ and would spuriously fail). An arm that Eb-drains to momentum *without* firing
   (θ_max < 0.9) is a miss (the over-boost mode).
5. **El-Badry overlay** (cheap, offline; NOTE it evaluates OUTSIDE W by construction — his
   validity needs t ≳ 3 Myr, past the SNe-safe cap — so it runs on the diagnostic arms'
   full-length θ(t) and is **flag-only, never a pass/fail bar**): θ(t) vs
   θ_EB(λδv=3, n_amb(R2(t))) via the `make_elbadry_theta.py` machinery; expect ±0.1 agreement
   where n_amb ∈ [1e2, 1e5] (bench-1's n̄=43 sits below the floor — exclude it).
6. **Report the Lcool/Lleak split** alongside every θ (Rogers & Pittard 2013: in porous 3D
   clouds 60–75% of wind energy *leaks* rather than radiates — a θ match with the wrong channel
   split would be a false positive).

**Prediction registered (lit agent):** the whole-band f_A from Phase 4 should land in
**[8, 13]** by direct inversion (θ₀ range 0.51–0.72, exponent 0.30), or ≈12.7 via the f_mix
equivalence (4^{0.55/0.30}) — and the same value should put every benchmark inside the Θ_cum
band. **Honesty (circularity):** calibrating f_A on L21b and then "agreeing" with L21b is a fit,
not a validation — the independent support is El-Badry's √n shape, the CEM ~25% precedent, and
the Eq-47 ṁ trend from Phase 4. Say exactly that in the paper.

Artifacts: `runs/params/bench5/`, `runs/data/bench5_summary.csv`,
`data/make_bench5_analysis.py`, `bench5_theta_tracks.png`; CONTAMINATION register entries;
REPRODUCE rows.

### Phase 6 ⬜ — decision (pre-committed tree; don't relitigate)

| outcome | verdict | action |
|---|---|---|
| single f_A fires 7/7 fireable + controls unchanged + benchmarks pass Phase 5 metric 1 | **f_A is the production-candidate successor** | maintainer ruling on default flip; paper narrative: physical knob calibrated on L21b, f_mix = its frozen-structure limit |
| 7/7 but benchmarks miss low (Θ_cum < 0.9) | knob right, magnitude short — likely the 1D leak/geometry gap | keep f_mix production; publish f_A + benchmark gap as the honest fidelity statement |
| 7/7 only at high f_A (≈24–32) and benchmarks overshoot there (Θ_cum > 0.99 or Eb-drain misses) | the single-scalar compromise fails calibration: band-fire and benchmark-match want different f_A | keep f_mix production; next rung = the state-coupled f_A (§4 — density dependence emerges, one value no longer needs to serve all clouds) |
| no whole-band f_A within grid ∪ bracket | calibrated per-config knob (like κ but honest fates) | keep f_mix; f_A becomes the fidelity/appendix knob; record windows |
| dense arms condense-first across the grid | the McKee–Cowie edge IS the dense transition | acceptable physics (2026-07-02 ruling precedent); document band-fire + condense-handoff split |
| freezes / no-root grind | P3 falsified live | full stop; freeze-watch instrumentation (`KAPPA_FREEZE_MECHANISM.md §5`); do NOT tune around it |
| p_source ≪ 2 or ≫ 4.5 | screen exponent didn't survive coupling | re-derive the response on live arms; update §2 reading and the f_A↔f_mix mapping |

## 4. Deferred track (separate workstream — do not start from this doc)

The **generalized near-front IC**: κ(T)·dT/dx = q_w + c_p·F_ṁ·(T−T_w), where q_w is the
conductive flux into the wall (lumping sub-anchor radiation — the same neglect Weaver Eq. 44
makes, upgraded from "none" to "lumped"). dR2 and the anchor gradient become one quadrature for
ANY κ(T):

```
dR2 = ∫_{T_w}^{T_init}  κ(T) / (q_w + c_p·F_ṁ·(T − T_w))  dT ,   F_ṁ = ṁ/4πR2², c_p = (5/2)k_B/μ
dT/dr|_anchor = −(q_w + c_p·F_ṁ·(T_init − T_w)) / κ(T_init)
```

Four payoffs: recovers Weaver exactly (q_w=0, T_w=0 — verified against
`bubble_luminosity.py:380–388`); fixes the κ_mix boundary divergence (quadrature converges for
κ_mix ∝ 1/T once T_w = 1e4 > 0); admits the saturation cap; and **stays defined for ṁ < 0**
whenever q_w exceeds the advected enthalpy release — the condensation branch (fix #4) as an
increment, with Weaver (x^{2/5}) and El-Badry (x^{2/7}, his Eq. 44) as the two poles indexed by
q_w. Closure: q_w ≈ f_A·L₃/(4πR2²) on the previous iterate. Plus the efficiency items: the
3-unknown joint root-find (β, δ, dMdt) to flatten the nested iteration (§8d cliff), and the
Kirchhoff variable y = ∫κdT (Marshak-standard; regularizes the front, cuts LSODA
micro-stepping). Each carries the full rule-5 ladder. **The L2 wiring is the state-coupled
plumbing**: upgrading scalar f_A → f_A(R2, Pb, λδv, T_pk) (El-Badry L_int closed form,
`ELBADRY_REFERENCE.md §7`) swaps one read at the same two sites; rerun Phases 3–5.

## 5. Citations (for the eventual paper section; verification flags per the 2026-07-06 sweeps)

Core: Spitzer 1962; Weaver+77 (ApJ 218, 377); Cowie & McKee 1977; McKee & Cowie 1977; Dalton &
Balbus 1993; Mac Low & McCray 1988 (the C=6e-7). f_κ precedent: Gupta+18 (MNRAS 473, 1537).
Interface physics in 1D: El-Badry+19 (MNRAS 490, 1961). Turbulent regime: Fielding+20 (ApJL 894,
L24); Tan, Oh & Gronke 2021 (MNRAS 502, 3179); Lancaster+21a/b (ApJ 914, 89/90), 2021c (ApJL
922, L3), 2024 (ApJ 970, 18), 2025 CEM (arXiv:2505.22730/22733). Suppression bounds: Markevitch
& Vikhlinin 2007; Chandran & Cowley 1998; Narayan & Medvedev 2001. Leakage counter-benchmark:
Rogers & Pittard 2013 (MNRAS 431, 1337). Consistency-only (never fit targets): Fierlinger+16;
Geen+21/23. Numerical-mixing caution for 3D-calibrated factors: Gentry & Krumholz 2019.
⚠️ Unverified-number flags: Weaver appendix eq. numbers/near-edge coefficient; CM77 saturation
exponent (−5/6 vs −5/8); El-Badry exact κ₀; **L21b Table-1 details (M_*, V_w, M–R pairings) —
abstract/search-level only; PDF-verify before Phase 5** (this environment's proxy blocks
non-GitHub fetches; [V] there means search-snippet confirmation).

## 6. Sibling reconciliation (keep true on every edit)

- `FA_IMPLEMENTATION_SPEC.md` — **deleted 2026-07-06** (folded into §3). Historical references
  (PLAN ledger 2026-07-06 entry, commit `bcf778a`) mean this doc.
- `FINDINGS.md §15` (the screen) + `§15a` (Phase 1, when run) + `§16` (the latent fallback
  double-boost flag).
- `INDEX.md` §2 table + §3 live thread — single-row registration pointing here.
- `PLAN.md` ledger — dated entries.
- `CONTAMINATION.md` — `fA_source_boost{,_summary}.csv` registered; future rows: edge map,
  theta5s, bench5.
- `REPRODUCE.md` #36 (screen); add rows as Phase 1/4/5 artifacts land.
- `KAPPA_EFF_SCOPING.md §6.2`, `KMIX_SELFCONSISTENT.md §2/§3`, `KAPPA_FREEZE_MECHANISM.md §7` —
  pointer patches (landed 2026-07-06).
- `LANCASTER_REFERENCE.md` — owes the Table-1 verification (Phase 5 pre-step).
