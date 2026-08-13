# PLAN — fixing the P_HII identity (branch `bugfix/phii-pt1`)

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

**Status (2026-08-12):** 🔵 actionable — **Batches 0, 1 and 4a are DONE; D1/D2 answered, and the
diagnosis has moved.** The momentum `P_HII + P_ram` sum is *intended* (D1) — so **C1 is rejected**;
the defect is the circularity `P_ram → Pb → shell_n0 → n_IF_Str → P_HII`, not the sum. §3b proves
the **cap is only the last link**: `ΔV ∝ shell_n0^-2.13` already forces `n_IF_Str ∝ shell_n0^+1.04`
(r = 0.993, N = 803) *before* the cap, and 4a's uncapped runs never put `P_HII` below `Pb`. The
intervention point is the shell ODE's inner BC (`shell_structure.py:124-126`), not the cap (`:251`).
**Next: Batch 5 (C3), starting with an offline screen of the decoupled formulations.** Prior
findings retained below. The cap
binds on **100% of rows in every phase**, and the blow-up it suppresses tops out at **3.33×**, so
the pre-registered C2a kill bar (p99 > 1e2) did **not** trip. **Batch 4a then removed the cap and
measured it: it survives cleanly on 4/4 configs — no numerical distress, no fate changes, faster
than baseline — but shifts trajectories 15.3–28.4%, over the 5% bar on every config.** So bare
removal is *safe but not neutral*, and the open question is now physics, not numerics (**D2**). The only `trinity/` change so far is the inert shadow diagnostic `n_IF_Str_raw`
(bit-identity gated). Two Batch-0 by-products changed the problem statement — the double-count is
far larger than assumed (transition median **1.82×**, momentum exactly **2.000×**), and the energy
phase carries a *separate* defect: `P_HII` smuggles the un-ramped bubble pressure past the
`dt_switchon` R1 ramp (up to **3.2×**). Evidence base: `docs/dev/phii-identity/README.md`.
Branch `bugfix/phii-pt1`.

---

## 0. The contract: one source of truth, self-updating, contamination-free

This section is normative. It exists because the transition/pdv-trigger effort sprawled to
10+ docs that drifted against each other, and because this fix touches ODE right-hand sides where
a contaminated comparison produces confident nonsense.

**One doc.** Everything about the fix effort lives HERE: strategy, candidates, gates, batch
results, verdicts, decisions, the dated log. Do **not** create sibling `FINDINGS.md`,
`RUNBOOK.md`, or per-batch notes — results land in this doc's §8 ledger and §9 log. The only
other files this workstream may grow are committed artifacts (`data/*.csv`, `figures/*.png`) and
runnable code (`harness/*`), each indexed in §8.3. The sibling `README.md` stays what it is — the
frozen-ish evidence record; it gets a pointer here and per-visit reconciliation, nothing more.

**Self-update protocol.** Every visit, in order:
1. Re-verify the §2 line references and §4 param paths against current source; fix drift in place.
2. Update batch Status fields (§6) and ledger tables (§8) in place — the tables are the state.
3. Append a dated entry to §9 (what changed, what was learned, what was re-planned and why).
   No entry, no edit — an undated change is contamination of the record.
4. Re-rank the remaining batches if the new evidence warrants it; record the re-ranking in §9.
5. Supersede by marking ⛔ with a date and one-line reason. Never delete history.

**Contamination rules** (all mandatory; violations invalidate the batch):
- **C-1 Pinned baseline.** Every comparison names its two git SHAs. The effort's base is
  `6d84b1e` (= `main` @ `731ac50` + the evidence workstream). If any of the three sibling
  branches (`feature/threeway-pt2`, `feature/low-winds-regime`, `hotfix/other-magic-numbers`)
  merges into `main` mid-effort, STOP, re-baseline Batch 0 on the new `main`, and log it.
- **C-2 No cross-branch code.** The three sibling branches are read-only evidence. Do not port
  their harness code or (worse) their `trinity/` edits piecemeal; lift ideas, reimplement here.
- **C-3 Separate processes.** One run per process, always (trinity leaks module-level global
  state in-process — CLAUDE.md rule 5). `run.py --workers N` satisfies this; calling the solver
  twice in one Python process does not.
- **C-4 Single param source.** Arms differ by code ref or by exactly one toggle line, never by
  hand-copied param files. A harness that materializes a variant param writes the diff into the
  artifact header.
- **C-5 Matched `t`.** Runs truncate at different `t`; every trajectory comparison interpolates
  to matched simulation time and reports the compared window (the screen tool does this — §5).
- **C-6 Provenance stamps.** Every committed CSV starts with `# generated <UTC> | <command> |
  code <SHA>[+dirty]`. No stamp, no trust.
- **C-7 Never reuse outputs across code changes.** Output dirs embed arm + SHA in the name;
  a `dictionary.jsonl` produced by different code than the header claims is poison.

## 1. Problem statement (condensed from README.md — the evidence doc)

Wherever the Strömgren-density cap binds, `P_HII` is an exact algebraic relabelling of the
confining pressure: `shell_n0` is defined by pressure balance against `Pb`
(`shell_structure.py:124-126`), `n_IF_Str` is capped at it (`:251`), and `P_HII` converts back
with the same three factors (`run_{energy,transition,momentum}_phase.py:224/564/634`). Five
sightings across three branches, 4–10 digits, ULP-level residual reproduced by
`harness/roundtrip_ulp.py`. **There are two distinct defects, and they are orthogonal:**

- **D-identity (the cap):** while the cap binds, `P_HII` carries no information about `Qi`,
  `f_esc`, or the ionized volume. The "photoionized gas" channel is fictional in that regime.
- **D-sum (the ODE forms):** transition drives on `max(Pb, P_HII + P_ram)` — which, with
  `P_HII ≡ Pb`, reduces to `Pb + P_ram` on every step (the `max` never binds) — and momentum
  drives on the bare `P_HII + P_ram = 2·P_ram`. Both are ODE right-hand sides; fates are
  downstream.

Phase exposure: 1a/1b safe (`max(Pb, P_HII)` absorbs the identity exactly); 1c and 2 affected.

⚠️ **Superseded 2026-08-12 by Batches 0/1 — read this paragraph with the corrections below.**
1. **There is no cap-slack window.** The cap binds on 100% of rows in every phase of every config
   measured. weak-winds' `Pb/P_HII = 0.33` at t=0 is an artifact of reconstructing `Pb` from
   `F_ram` (which carries the *ramped* pressure); reading `Pb` directly gives 1.0000000000 there.
   The few rows where `P_HII ≠ Pb` are `Pb` staleness at the 1a→1b handoff, with the cap still bound.
2. **D-sum is much bigger than "a few percent".** Transition's median `P_drive` overshoot is
   **1.82×**, reaching 1.998; momentum is exactly **2.000×**.
3. **A third defect exists, in the phase this doc called safe.** 1a/1b remain safe *in the `max`
   sense* — `P_drive == Pb == P_HII` — but `get_effective_bubble_pressure` applies a
   `dt_switchon = 1e-3` Myr ramp that pulls R1 → 0, while `params['Pb']` (hence `P_HII`) uses the
   un-ramped R1. Inside that window the two differ by up to **3.2×**, so the `P_HII` channel
   reintroduces exactly the pressure the ramp exists to suppress. **This is the single biggest
   risk to any cap fix**: removing the cap drops early driving pressure by up to 3×, which will
   look like the fix breaking the code. Call this **D-ramp**; it is new work, tracked in §9.

## 2. Maintainer input on record

**2026-08-12 (this session):** the cap's origin is numerical, not physical — *"Originally i had
the PHII cap because at small volume it'd give very high n_str_if and that would give very high
PHII, and i dont know if that breaks things."* At small ionized volume ΔV → 0 the Strömgren
balance `n_IF_Str = sqrt(3(1−f_esc)Qi / (4π χ_e αB ΔV))` (`shell_structure.py:246`) diverges;
the cap was the guard.

This materially updates the evidence doc's §7.2: the cap is **not** a deliberate physics claim
that HII pressure can never exceed the confining pressure — it is a blow-up guard whose side
effect is the identity. Consequence for strategy: replacing the guard with one that doesn't
manufacture the identity is on the table (§3, C2b), and "the sum is intended because the cap is
intended" is not a valid inference. The intent question for the *sum* (evidence doc §7.1) is
still open and is decision **D1** below.

**2026-08-12 (D1 + D2 answered).** Maintainer rulings, verbatim in substance:

- **D1 — the momentum sum is INTENDED.** `P_drive = P_HII + P_ram` is the intended momentum-phase
  form. *But* the maintainer's condition is that `P_HII` "should be its own calculation and be
  decoupled from `P_ram` as much as possible", because today the chain is circular:
  `P_ram → Pb → shell_n0 → n_IF_Str → P_HII`. So the sum is not the defect; **the circularity is**.
  C1 (`max` instead of `+`) is therefore **REJECTED for the momentum phase** — it fixes the wrong
  thing. ⛔ 2026-08-12.
- **D1 (transition) — the `max` is deliberate.** `max(Pb, P_HII + P_ram)` is intended as a gradual
  handover between the thermal and momentum drives as `Pb → 0`. Maintainer is open to a better
  formulation but has not proposed one. So the transition `max` is NOT to be "fixed" into a
  competition; any replacement must still be a smooth handover.
- **D2 — `P_HII` should be a real, separate pressure**, and should be treated as one unless the
  architecture genuinely cannot support it (in which case the assumption must be explicit).

Still needed from the maintainer: D3, D4 in §7.

## 3. Fix candidates

Legend: footprint = phases whose dynamics can change. All candidates keep `include_PHII`
working as a global off-switch.

| id | change | footprint | risk | cost |
|---|---|---|---|---|
| **C0** | none — reference arm | — | — | free |
| **C1** | *transmit, don't add*: momentum `P_drive = max(P_HII, P_ram)` (`run_momentum_phase.py:265,445`); transition `P_drive = max(Pb, P_HII, P_ram)` (`run_transition_phase.py:331`, `energy_phase_ODEs.py:253,385`) — 5 expression sites | 1c, 2 only; **1a/1b must be bit-identical** (hard gate) | low — smallest diff that kills D-sum | tiny |
| **C2a** | *bare cap removal* (`shell_structure.py:251` deleted) | ALL phases (1a/1b via `max(Pb, P_HII_raw)`, which un-absorbs whenever raw > `shell_n0`… i.e. exactly where the cap used to bind) | **high** — the ΔV→0 divergence the cap was built for; interacts with the per-segment freeze ratchet (phase1a-init Extra finding #1) which is *catastrophic* at compact scale | tiny diff, expensive validation |
| **C2b** | *replace the guard*: keep a blow-up guard that is not the confining pressure — candidates: floor ΔV at a resolved skin thickness; cap at `shell_nMax` instead of `shell_n0`; smooth (harmonic) min | ALL phases, but only where the old cap bound AND the new guard differs | medium | small |
| **C3** | *advanced physical method* (needs D2 + a design pass): **(a) interface-pressure transmission** — drive the neutral shell with the pressure at the ionized/neutral interface taken from the already-integrated ionized-layer structure (`shell_structure.py` integrates `nShell_arr_ion(r)` with radiation), so no density→pressure back-conversion exists and no cap is needed; **(b) two-zone regime switch** — classical D-type (Spitzer-like) expansion when `P_HII_raw > P_ram`, wind-driven otherwise (crude limit of this = C1⊕C2); **(c) excess-only partition** — `P_drive = P_ram + max(P_HII_raw − P_confining, 0)`: the skin transmits the confining pressure and only its *excess* adds | ALL | design risk; must reproduce limiting cases (wind-only → Weaver-like, photo-only → Spitzer-like) | largest |
| **C4** | diagnostics-only honesty fix: report `F_HII` as the independent component, not the relabelled one | none (output only) | none | tiny; fold into whichever lands |

**Composability — read this before picking arms.** C1 and C2 attack different defects:
C1 fixes D-sum but leaves `P_HII` fictional while capped; C2 fixes D-identity but leaves the
double-count arithmetic in place (merely no longer *exactly* 2×). They compose: **C1⊕C2 =
`max(P_HII_raw, P_ram)`**, which is also C3b's crude limit. C3c *requires* C2 (with the cap,
excess ≡ 0 identically). So the batches measure C1 and C2 separately first — their composition
is then predictable rather than a third experiment.

## 3b. The circularity is NOT the cap — measured 2026-08-12

D2 asks for a way to decouple `P_HII` from `P_ram`/`Pb`. Batch 4a plus the b1 raw diagnostic locate
the coupling precisely, and it is **not** where the workstream assumed.

**The cap is only the last link.** With the cap removed (Batch 4a, self-consistent runs) the ratio
`P_HII/Pb` is still **1.06–3.55 and never below 1** on any row of any config. Cap removal changes
`P_HII` from `1.00·Pb` to `(1–3.5)·Pb`; it does not decouple it.

**The real coupling runs through the ionised volume.** Regressions over the b1 arm
(N = 803 rows, 8.8 decades of dynamic range, stock dynamics + pre-cap diagnostic):

| relation | measured | meaning |
|---|---|---|
| `shell_n0` vs `Pb` | exact (`shell_structure.py:124-126`) | the inner BC *is* `Pb/(k T)·μ` |
| `log ΔV` vs `log shell_n0` | slope **−2.126**, r = −0.909 | denser shell ⇒ thinner ionised skin |
| `n_IF_Str ∝ ΔV^(−1/2)` | by construction (`:246`) | — |
| ⇒ `log n_IF_Str_raw` vs `log shell_n0` | slope **+1.039**, r = **0.993** | `P_HII` tracks `Pb` linearly |

−2.126 × (−1/2) = +1.06, which is the +1.039 measured. The chain closes quantitatively:

```
Pb ──> shell_n0 ──> [shell ODE inner BC] ──> R_IF ──> ΔV ∝ shell_n0^-2.13
                                                        │
                          n_IF_Str ∝ ΔV^-1/2 ∝ shell_n0^+1.06 ≈ Pb  <──┘
```

**So the intervention point is `shell_structure.py:124-126`, not `:251`.** The Strömgren balance is
being evaluated over a volume whose size is set by `Pb`, so it cannot report anything but `Pb`. Any
decoupling must break the ΔV path; removing the cap alone provably does not.

### C3 candidates, re-derived against this diagnosis

- **C3a — Strömgren over the cavity.** `n_HII = sqrt(3 Q_i,abs / (4π χ_e α_B R2³))`: depends on
  `Qi` and `R2` only, **zero** dependence on `Pb`, `P_ram` or `shell_n0`. Measured offline on B3M's
  momentum rows: n = 235 → 49 cm⁻³ and P/k = 2.4e6 → 4.9e5 K cm⁻³ over R2 = 6.6 → 19 pc — physically
  reasonable H II magnitudes. Scaling `P_HII ∝ R2^-3/2` vs `P_ram ∝ R2^-2` means the two **cross
  over**, which is the coevolution behaviour D2 asks for. ⚠️ Caveat: it assumes ionised gas fills
  the cavity, which sits awkwardly with the wind-evacuated-cavity picture, and on B3M it makes
  `P_HII` ≈ 5–7× `P_ram`, i.e. it would dominate the momentum drive.
- **C3b — pre-shock/ambient Strömgren.** Set the balance from the unperturbed cloud density at `R2`
  (a pure input from the density profile). Decoupled by construction; the classical D-type front
  picture. Untested.
- **C3c — decouple the skin's inner BC only.** Keep the current skin geometry (which *is* the right
  place for the ionised layer physically — between the wind and the neutral shell) but stop letting
  `Pb` set the ionisation calculation's starting density. Smallest conceptual change, hardest to
  specify; needs a defensible independent thickness or density scale.

**Recommended next (Batch 5): C3a and C3b measured offline first**, on committed b0/b1 snapshots —
no simulation needed, because both are closed-form in quantities already stored (`Qi`, `R2`,
`rCloud`, profile). Only the surviving candidate gets a run arm. This inverts the usual order
deliberately: the cheap screen is decisive here because the failure mode is *magnitude*, not
stability.

## 4. Config / regime matrix

All committed, all on this branch. Two tiers: **core-6** gates every batch; **full-12** gates
landing (Batch 6). Wall times unknown until Batch 0 measures them (`wall_s` column in the
ledger); prior anchors: screen 5-config two-arm ≈ 1 h at `stop_t 0.02` (screen/README), the
three bench momentum A/B ≈ 30 min (momentum-pdrive), weak-winds control ≈ 35 min at
`stop_t 1.5`.

| tier | id | param | regime it exercises |
|---|---|---|---|
| core | SC | `param/simple_cluster.param` | energy-driven baseline (CLAUDE.md's named edge) |
| core | F1LO | `docs/dev/performance/f1edge_lowdens_himass_hisfe.param` | low density × high mass × high sfe — feedback-strong edge |
| core | F1HI | `docs/dev/performance/f1edge_hidens_himass_losfe.param` | high density × high mass × low sfe — stiffest committed edge |
| core | B3M | `docs/dev/transition/pdv-trigger/runs/params/bench5/bench3_m1e5_r5__none_diag.param` | momentum-heavy; 104 momentum rows, 88× `P_ram` range — the sharpest identity arm |
| core | PRB | `docs/dev/phase1a-init/harness/params/probe.param` | compact sub-GMC — small-ΔV blow-up regime + freeze-ratchet stress; **C2's likely failure point** |
| core | WW | SC + `FB_thermCoeffWind 0.1` (harness-materialized per C-4) | weak winds — cap-slack candidate; where an independent `P_HII` floor matters most (weak-winds H2) |
| full | B1M | `…/bench5/bench1_m5e4_r20__none_diag.param` | momentum, gentler (1.9× range) |
| full | B2M | `…/bench5/bench2_m1e5_r10__none_diag.param` | momentum, mid (22× range) |
| full | GMC | `docs/dev/phase1a-init/harness/params/gmc_control.param` | GMC-scale control for PRB |
| full | BE | `docs/dev/transition/cleanroom/configs/be_sphere.param` | Bonnor–Ebert profile (slope axis) |
| full | PL2 | `docs/dev/transition/cleanroom/configs/pl2_steep.param` | steep power-law profile (slope axis) |
| full | LDLS | `docs/dev/transition/cleanroom/configs/large_diffuse_lowsfe.param` | low density × low sfe — the one cleanroom config whose `Eb` actually peaks |
| full | SDHS | `docs/dev/transition/cleanroom/configs/small_dense_highsfe.param` | high density × high sfe, compact |

Axes covered: density 4+ decades (F1LO↔F1HI, LDLS↔SDHS), mass (5e4→himass), sfe (losfe↔hisfe),
profile slope (PL0/BE/PL2), phase chronology (energy-only SC ↔ momentum-heavy benches ↔ compact
PRB), wind strength (WW). This is the "high/low dens, high/low mass, sfe, slopes, pdv/compact"
spread, drawn from the params the prior workstreams already trust.

## 5. Measurement infrastructure

- **Comparator:** `docs/dev/screen/screen.py` — two git refs × N configs, separate worktrees +
  processes, matched-`t` ledger, fate check, exit-1 gating. Built for exactly this ("run it
  before landing a scheme change"); it has never been run in anger, so Batch 0 doubles as its
  first real outing. Extend its `CONFIGS` map with §4 (a screen-tool change, kept out of
  `trinity/`, gated by `test/test_scheme_screen.py`).
- **Identity harvester:** new `harness/harvest_identity.py` — reads a run dir's
  `dictionary.jsonl`, emits per-phase identity metrics (relΔ `P_HII` vs confining pressure,
  cap-binding fraction, raw/cap blow-up ratio once Batch 1 lands) with C-6 stamps. Reimplements
  the useful third of momentum-pdrive's `check_phii_pram.py` under C-2.
- **Bars:** trajectory attention bar `|ΔR2/R2| > 5%` at matched `t` (the G2 bar phase1a-init
  adopted 2026-08-05); identity bar relΔ ≤ 5e-16 (measured ceiling 3.6e-16 + margin); bit-identity
  bar = byte equality of `dictionary.jsonl` (Batch 1: after stripping the new key from each row).
  Fate changes are **enumerated, never silently passed** — under C1 a fate flip may be the *fix
  working*; the gate is "explained and signed off (D3)", not "unchanged".

## 6. The batch ladder

Statuses: ⬜ not started · 🔷 in flight · ✅ pass · ❌ fail · ⛔ superseded. Every batch: bars
pre-registered here *before* it runs (edit bars only via a dated §9 entry, never retroactively);
artifacts committed same session; §8 ledger updated; full `pytest` + ruff F-rules before any
commit that touches code.

### Batch 0 — baseline capture + first identity grid — Status: ✅ PASS (2026-08-12)
No code change. Run the **full-12** at base SHA (separate processes); harvest trajectories +
force budgets + identity metrics; record wall times.
- Commands: `python run.py <param>` per config (or a sweep param + `--workers 4`), then
  `python docs/dev/phii-identity/harness/harvest_identity.py outputs/<run>...`
- Artifacts: `data/b0_trajectories.csv` (matched-grid, one block per config),
  `data/b0_identity_grid.csv` (per config × phase), wall-time column.
- **PASS bars:** ≥ core-6 exit cleanly (full-12 failures documented, not fatal); identity bar
  holds wherever the cap binds — this *extends the evidence to the full grid* and re-derives the
  three bench numbers as a consistency check (they must reproduce to the printed digits).
- Kill/branch rule: a config that won't run at base is dropped from the matrix by dated log
  entry, not silently.

### Batch 1 — shadow uncapped diagnostic (the C2 de-risk) — Status: ✅ PASS (2026-08-12)
Smallest possible `trinity/` change, diagnostics-only: `shell_structure.py` also returns the
**pre-cap** value as `n_IF_Str_raw` (pattern: `n_IF_ODE` at `:225` — a raw value already kept for
diagnostics), new `ParamSpec` (`runtime_shell`, like `registry.py:513-515`), so it lands in every
snapshot.
- **PASS bars:** (i) dynamics bit-identical on core-6 — every pre-existing key of every matched
  row byte-equal; (ii) full `pytest` green; (iii) product delivered: the **cap-binding map** —
  per config × phase: fraction of snapshots with raw > `shell_n0`, and the blow-up distribution
  `raw/shell_n0` (max, p99).
- Artifacts: `data/b1_capmap.csv`, updated `b0_identity_grid` columns.
- **Pre-registered C2 kill bar:** if p99(`raw/shell_n0`) > 1e2 in any core config's phase 1a/1b,
  **C2a is dead on arrival** (the ODE would see a 100× pressure spike exactly where the freeze
  ratchet amplifies it) — skip to C2b/C3 without running Batch 4a. This answers "i dont know if
  that breaks things" *predictively*, for the cost of one diagnostics run.
- Also closes evidence-doc §7 open question 4 (where does the cap bind?) with data.

### Batch 2 — zero-code bracket: `include_PHII` off vs on — Status: ⬜
No `trinity/` change; the knob exists (`registry.py:365`). Screen the matrix `False` vs `True`.
- Why it brackets: `P_HII = 0` ⇒ momentum drives on `P_ram`, transition on `max(Pb, P_ram)` —
  **identical to C1 wherever the cap binds**. Differences from C1 localize to cap-slack windows
  (early 1a, where `P_HII = 3·Pb` genuinely drives, and late times). So B2 ≈ a free preview of
  C1, plus the maximal envelope any P_HII fix can reach.
- **Confound, documented up front:** the off arm also removes the *legitimate* early-1a HII
  driving; interpret early-time ΔR2 as envelope, not as C1 prediction. B1's cap map says exactly
  where the confound lives.
- **PASS bars:** screen completes on core-6; ledger records ΔR2(t), Δfate, Δt_end per config;
  no bar on the *size* of Δ (this batch measures, it does not judge).
- Artifacts: `data/b2_bracket_ledger.csv`.

### Batch 3 — C1: transmit-don't-add — Status: ⬜
Implement the 5-site diff (§3). Gates, all mandatory:
- **G3.1 (hard):** phases 1a/1b **bit-identical** full-run on core-6 (the diff's transition
  branch lives in the shared `energy_phase_ODEs.py` — this catches a slipped guard).
- **G3.2:** screen vs C0 on core-6 at matched `t`; every |ΔR2| > 5% and every fate change
  enumerated with the phase it originates in (must be 1c/2 only).
- **G3.3 (mechanism cross-check):** wherever B1 says the cap binds, C1's transition/momentum
  trajectories must match B2's off-arm to tight tolerance (they are algebraically identical
  there). Divergence = implementation bug, not physics.
- **G3.4:** full `pytest`; goldens re-baselined only under D4 sign-off, with the before/after
  values recorded in §8.
- Artifacts: `data/b3_c1_ledger.csv`; the diff itself.

### Batch 4 — C2: cap experiments — Status: 🟡 **4a DONE (survives, over bar); 4b not started**
- **4a (bare removal, C2a):** only if the kill bar did not trip. Screen vs C0 across core-6 +
  PRB compulsory. Watch: integrator stalls, the bubble-structure monotonic guard, overflow;
  `P_HII` decoupling from `Pb` (the identity must *break* — that's the point); freeze-ratchet
  amplification at PRB.
  - **Arm:** the cap line (`shell_structure.py:251`) commented out in a throwaway git worktree, so
    the main tree stays clean and the b1 WW run could proceed concurrently. Runs land under
    `outputs/phii/b4a__088a8d6_dirty/`. The edit is deliberately NOT committed to the branch —
    4a is a measurement, not a proposed change.
- **4b (guard replacement, C2b):** pick ONE replacement guard by B1's data (the one that binds
  only in the blow-up regime and nowhere else), screen it identically.
- **PASS bars:** run survival on all arms; identity broken (relΔ `P_HII` vs `Pb` becomes O(1)
  where the old cap bound); ΔR2/fate ledger complete; PRB terminates.
- Artifacts: `data/b4a_ledger.csv`, `data/b4a_identity_grid.csv`.

**4a result (2026-08-12).** Every PASS bar met except the trajectory bar, which is breached everywhere:

| config | wall_s | fate base → 4a | ΔR2 max | at t | ΔR2 end | distress |
|---|---|---|---|---|---|---|
| PRB | 764 | stopping_time → stopping_time | 28.4% | 1.3e-07 | **0.95%** | none |
| B3M | 625 | stopping_time → stopping_time | 27.4% | 9.0e-06 | 13.6% | none |
| F1LO | 762 | stopping_time → stopping_time | 25.9% | 1.4e-04 | 14.4% | none |
| F1HI | 492 | shell_collapsed → shell_collapsed | 15.3% | 3.2e-06 | 6.2% | none |

- **Survival: unambiguous.** No stalls, no overflow, no monotonic-guard rejection, no convergence
  warnings, and 4a is *faster* than baseline on every config. The ΔV→0 blow-up the cap was built
  for does not materialise in any regime tested — including PRB, the compact probe chosen precisely
  because it is the most likely place for it.
- **Identity destroyed, as intended.** `frac_PHII_eq_Pb` = 0.0000 in every phase of every config.
  `P_HII` again depends on `Qi`, `f_esc` and the ionized volume.
- **But it is a real physics change, not a cleanup.** Uncapped `P_HII` exceeds `Pb` on 100% of rows
  (up to 3.36×), so it now *wins* `max(Pb, P_HII)` everywhere and the median `P_drive/Pb` rises
  1.0000 → 1.83 (PRB). Every ΔR2 maximum sits inside the `dt_switchon = 1e-3` Myr window: the cap
  was doing its heaviest work exactly where the R1 ramp is active.
- ⚠️ **Retraction (2026-08-12).** An earlier reading of the §1 correction predicted cap removal would
  *lower* early driving pressure (by removing `P_HII`'s un-ramped-pressure smuggling). That is
  backwards and the data says so: the cap clamps the Strömgren density *down*, so removing it raises
  `P_HII` above `Pb` and the drive goes **up**. Recorded so the wrong prediction is not re-derived.
- **Open, and it is now the crux:** the larger uncapped `P_HII` is only trustworthy if the Strömgren
  balance is trustworthy at these ionized volumes. Nothing measured here settles that — it is D2.

### Batch 5 — C3: the advanced method — Status: ⬜ (only if B3/B4 fail their gates, or D2 asks for it)
Not designed here beyond §3's three candidates — a design pass goes THROUGH this doc (new §,
dated) and needs D2 first. Pre-registered acceptance floor for any C3 design: reproduces the
wind-only limit (matches C0 when `Qi → 0`) and the photo-only limit (Spitzer-like `R ∝ t^{4/7}`
slope when `Lmech → 0`, checked on a WW-descendant config); then the full ladder as Batch 3.

### Batch 6 — land — Status: ⬜
Chosen candidate (D1 decides between C1, C1⊕C2b, or a C3) on the **full-12**; full ladder
re-verify; CHANGELOG entry; reconcile the evidence README (§7 answers), DOC_STATUS, and — when
the sibling branches merge — fold-back notes for momentum-pdrive (its §2 "inferred" caveat, its
CSV column rename) and weak-winds (quantitative collapse times now clean). Goldens re-baselined
under D4 with a table of before/after.

## 7. Decisions needed from the maintainer

| id | question | blocks | state |
|---|---|---|---|
| D1 | Is the `P_HII + P_ram` **sum** intended (separate reservoir) or is the skin a transmitter (→ `max`)? Evidence doc §5 shows the transition `max` never binds, so "it's already guarded" is not a defense. | Batch 3 landing (running/measuring it is not blocked) | **open** |
| D2 | ✅ **ANSWERED 2026-08-12** — `P_HII` should be a real, separate pressure, treated as one unless the architecture cannot support it (then the assumption must be explicit). Consequence: the target is **decoupling**, and §3b shows the cap is not the coupling — the ionised volume is. Open sub-question for Batch 5: which decoupled formulation (C3a/C3b/C3c) | Batch 5 | **answered; formulation open** |
| ~~D2-old~~ | ⛔ superseded by the above. **WAS THE CRUX (Batch 4a).** Removal is proven *safe* — no blow-up materialises in any regime tested, including the compact probe. So the question is no longer "can we?" but "should we?": is the uncapped Strömgren pressure physically trustworthy at these ionized volumes, given it exceeds `Pb` on 100% of rows (up to 3.36×) and shifts trajectories 15–28%? No measurement can settle this; it needs the model's intent. Also confirm §2's reading that the cap was pragmatic, not a physics claim. | Batch 4b design; Batch 5; **4a landing** | **open** |
| D3 | Fate flips under a candidate fix: acceptable-if-explained, or a re-tune trigger? | Batch 3/4 verdicts | **open** |
| D4 | Authority to re-baseline goldens (`test_phase_boundary.py`, `test_betadelta_hybr_stress.py`, `test_scheme_screen.py` fixtures) if the landed fix moves them. | Batch 6 | **open** |

## 8. Ledger (results land here — the one source of truth)

### 8.1 Batch verdicts
| batch | status | date | verdict (one line) | artifacts |
|---|---|---|---|---|
| 0 | ✅ | 2026-08-12 | **PASS** on 6/6 core. Identity holds on 100% of implicit/transition/momentum rows and ≥96.97% of energy rows, relΔ ≤2.9e-16, across 4 decades of nCore. B3M independently reproduces momentum-pdrive (`P_HII` vs `P_ram` = 2.39e-16 over 34 rows). Drive anatomy: implicit exactly 1, transition ≤1.998 (median 1.82), momentum exactly 2.000, energy ≤3.31. **`frac_nIFStr_eq_n0` = 1.0000 in every phase of every config** — the cap is bound everywhere, needing no diagnostic to show it | `data/b0_identity_grid.csv`, `data/b0_trajectories.csv`, `data/b0_walltimes.csv` |
| 1 | ✅ | 2026-08-12 | **PASS.** G1(i): B3M 231 + PRB 184 + WW 178 = **593 rows** exactly equal on every pre-existing key (repr compare), matching row counts ⇒ diagnostic inert; independently corroborated by the matched-t comparator returning 0.000% on both. Cap binds **100% of rows in every phase**; blow-up p99 1.06–3.33, max 3.33. **Kill bar NOT tripped ⇒ C2a survives, Batch 4a authorised.** Corrects B0: sub-100% energy rows are `Pb` staleness at the 1a→1b handoff, not cap-slack | `data/b1_bitidentity.csv`, `data/b1_capmap.csv` |
| 2 | ⬜ | — | — | — |
| 3 | ⬜ | — | — | — |
| 4 | 🟡 | 2026-08-12 | **4a MEASURED — survives, but is not behaviourally neutral.** 4/4 configs (PRB, B3M, F1HI, F1LO) ran to their natural end, **zero** distress lines (no excess-work, overflow, monotonic-guard or convergence warnings), wall times *within* baseline (492–764 s vs 682–832 s). **No fate changed** on any config. Identity destroyed as intended: `frac_PHII_eq_Pb` = **0.0000** in every phase of every config (was ≥0.9697), relΔ now O(1) (0.06–2.55). But **every config breaches the 5% bar**: ΔR2 max 15.3–28.4%, all located inside the `dt_switchon` window (t = 1.3e-7 … 9e-6 Myr); ΔR2 at end-of-overlap 0.95% (PRB, recovers) → 14.4% (F1LO, retained). Mechanism: uncapped `P_HII` exceeds `Pb` on **100%** of rows (max 3.36×) so it wins the `max`, lifting median `P_drive/Pb` from 1.0000 to 1.83 (PRB). **Verdict: C2a is numerically viable and physically consequential — not a free win. Landing it needs D2.** 4b not started | `data/b4a_ledger.csv`, `data/b4a_identity_grid.csv` |
| 5 | ⬜ | — | — | — |
| 6 | ⬜ | — | — | — |

### 8.2 Config wall-times (filled by Batch 0)
Shared 4-core box, 3–4 concurrent runs, so these are contention-inflated upper bounds.

| config | stop_t used | wall_s | snapshots | notes |
|---|---|---|---|---|
| SC | 1.5 (override) | 1369 | 192 | energy→implicit only |
| B3M | 1.5 (override) | 682 | 231 | **only config reaching momentum** — all 4 phases |
| F1LO | 1.5 (override) | 802 | 133 | |
| F1HI | 1.5 (override) | 832 | 144 | reaches transition (4 rows); fate `shell_collapsed` |
| PRB | 0.1 (as committed) | 713 | 184 | compact probe; 781 s standalone, uncontended |
| WW | 1.5 (override) | >5400 (killed) | 178 | timeout, but **reached its natural end**: collapse at t = 0.2816 Myr, v2 = −10 km/s, all 4 phases |

### 8.3 Artifact index
| file | producer | batch | stamp SHA |
|---|---|---|---|
| `data/phii_identity_evidence.csv` | (evidence phase) | pre | `6d84b1e` |
| `data/roundtrip_ulp.csv` | `harness/roundtrip_ulp.py` | pre | `6d84b1e` |
| `data/b1_bitidentity_ww.csv` | `harness/compare_bitidentical.py` | 1 | `088a8d6` |
| `data/b4a_ledger.csv` | `harness/compare_trajectories.py` | 4a | `088a8d6`+cap-removed |
| `data/b4a_identity_grid.csv` | `harness/harvest_identity.py` | 4a | `088a8d6`+cap-removed |
| `data/b4a_walltimes.csv` | `harness/run_batch.py` | 4a | `088a8d6`+cap-removed |
| `harness/b4a_cap_removal.patch` | the exact 4a code change (apply to `088a8d6` to reproduce) | 4a | `088a8d6` |
| `data/b0_identity_grid.csv` | `harness/harvest_identity.py` | 0 | runs @ `6b55657` |
| `data/b0_trajectories.csv` | `harness/harvest_identity.py` | 0 | runs @ `6b55657` |
| `data/b0_walltimes.csv` | `harness/run_batch.py` | 0 | runs @ `6b55657` |
| `data/b1_bitidentity.csv` | `harness/compare_bitidentical.py` | 1 | b0 `6b55657` vs b1 `bb302e0` |
| `data/b1_capmap.csv` | `harness/harvest_identity.py` | 1 | runs @ `bb302e0` |

Run dirs (not committed — regenerate with `harness/run_batch.py`):
`outputs/phii/b0__6b55657_dirty/`, `outputs/phii/b1__386df59_dirty/`. The `_dirty` suffix reflects
uncommitted **harness/docs** files at launch; `trinity/` was byte-identical to `6b55657` for every
b0 run and to `bb302e0` for every b1 run (§9 records how that was protected).

## 9. Dated log (append-only; newest last)

- **2026-08-12** — Plan created on `bugfix/phii-pt1` (base `6d84b1e` = `main` @ `731ac50` +
  evidence workstream). Recorded maintainer input on the cap's numerical origin (§2) from the
  live session; this reframed C2 from "remove a physics claim" to "replace a guard", added C2b,
  and put the B1 shadow diagnostic before any cap edit. Candidate set C0–C4 pre-registered with
  gates; matrix drawn entirely from committed params (screen defaults + bench5 + cleanroom +
  f1edge + phase1a-init probes + a WW rung). Nothing run yet.

- **2026-08-12 (later, same session) — Batches 0 and 1 ran; both PASS. Four things changed.**

  **(a) The identity is broader than recorded, and there is no cap-slack window.** Across five
  configs spanning 4 decades in `nCore`, 1e5–1e7 Msun, sfe 0.01–0.5 and compact-probe→GMC scale,
  `P_HII == Pb` to ≤2.9e-16 on **100%** of implicit, transition and momentum rows. Batch 1's
  `n_IF_Str_raw` then showed the cap binding on **100% of rows in every phase**, which reinterprets
  Batch 0's 96.97–99.24% energy figure: those rows are not cap-slack (the cap is still bound on
  them) but `params['Pb']` staleness — B3M has exactly one, at t = 3.0e-3 Myr, the 1a→1b handoff,
  a 7.2% offset. **Consequence:** §1's "cap-slack windows exist and matter" is retracted, and so is
  weak-winds' "P_HII is genuinely independent early and late". Their `Pb/P_HII = 0.33` at t=0 came
  from reconstructing `Pb` as `F_ram/4πR2²`; `F_ram` carries the *ramped* pressure (see (c)), so the
  reconstruction is off by exactly that factor. Read `Pb` directly and it is 1.0000000000.

  **(b) D-sum is an order of magnitude bigger than assumed.** The new `Pdrive_over_Fram_*` columns
  measure `P_drive` against the pressure implied by the reported `F_ram`. Transition's **median**
  is 1.824 (max 1.998) and momentum is exactly **2.000** — not the "~2.7%" a first read of the
  transition `P_ram/P_HII` ratio suggested (that ratio's 36.8 was a *max*, not typical). Implicit
  is exactly 1, confirming the `max` genuinely absorbs the identity there.

  **(c) A third defect, D-ramp — and it is the main risk to any cap fix.**
  `get_effective_bubble_pressure` (`get_bubbleParams.py:311+`) applies a `dt_switchon = 1e-3` Myr
  ramp pulling `R1 → 0`, while `params['Pb']` — which `P_HII` equals identically — uses the
  un-ramped `R1`. Inside that window the two differ by up to **3.2×** on all four configs
  measured, and the median outside it is exactly 1.000. So `P_HII` is silently reintroducing the
  pressure the ramp exists to suppress. Phases 1a/1b remain "safe" in the `max` sense
  (`P_drive == Pb == P_HII` on 149/150 SC rows) — but **removing the cap would drop early driving
  pressure by up to 3×**, which will read as "the fix broke everything" unless it is expected.
  This connects to `phase1a-init`'s stale-pressure ratchet and to the `dt_switchon` work on
  `hotfix/other-magic-numbers`. **Re-plan:** Batch 4 must A/B against a D-ramp-aware reference,
  not against C0 alone; and D1/D2 should be answered knowing the cap is currently load-bearing for
  the early-time pressure.

  *Method note:* (c) was nearly reported as a 3× **overdrive bug**. The first reconstruction used
  `Eb/(2πR2³)`, which is not `press_bubble`; checking `P_drive` directly showed
  `P_drive == Pb == P_HII` and the alarm dissolved. The real finding is the ramp mismatch. Retract
  fast when the data disagrees — CLAUDE.md rule 5.

  **(d) C2a is authorised.** Pre-registered kill bar: p99 `raw/shell_n0` > 1e2 in phase 1a/1b of
  any core config ⇒ C2a dead on arrival. Measured p99: **1.06–3.33**; max **3.33** (B3M energy),
  **3.31** on the compact probe — the small-ΔV regime the cap was built for. So the ΔV→0 blow-up
  the cap guards against does not materialise at anything like the feared magnitude in these
  regimes, and Batch 4a may run. *Caveat:* neither config probes the first instants where ΔV→0 is
  most acute, and PRB stopped at 0.1 Myr; a Batch 4a stall would still most likely appear there.

  **Process notes.** (i) *Contamination near-miss, caught:* the Batch 1 patch was applied while
  four Batch 0 runs were in flight. Each `run.py` imports fresh at spawn, so WW and F1HI — not yet
  started — would have run patched code inside the `b0__` tree, violating C-7. Verified by mtime
  that the four in-flight runs predated the edit, reverted `trinity/` until they finished, then
  re-applied. Rule C-7 needs a stronger form: **never edit `trinity/` while any run_batch stream is
  alive**, since a stream spawns its later configs long after launch. (ii) `run_batch.py` was
  clobbering its own wall-time CSV when driven as concurrent streams; now merges. Rows for the
  clobbered configs were recovered from the stream logs. (iii) The b0/b1 run dirs carry a `_dirty`
  suffix from uncommitted *harness* files; `trinity/` was byte-identical to `6b55657` (b0) and
  `bb302e0` (b1) throughout. (iv) WW (weak-wind rung) timed out at 5400 s having reached t ≈ 0.02
  of 1.5 Myr — the weak arm is slow in the implicit phase. It is the one core config without a
  Batch 0 row; re-run it with a shorter `stop_t` (≈0.3 Myr, past the weak-winds collapse at 0.282)
  rather than more wall time. (v) The `n_IF_Str_raw` ParamSpec shifted
  `test_materialize_runtime.py`'s pinned live-flow inventory 105→106 and its snapshot split 9/96→
  9/97; counts, test names and docstring provenance updated. Full suite: **1013 passed**.

- **2026-08-12 (Batch 0 completed to 6/6).** WW was re-read rather than re-run: its 5400 s timeout
  killed the *process*, but the run had already reached its physical end — **collapse at
  t = 0.2816 Myr, R2 = 0.897 pc, v2 = −10.05 km/s**, covering all four phases (78/53/20/27 rows).
  That independently reproduces weak-winds' headline fate flip (their smoke pair: SHELL_COLLAPSED
  at t = 0.282, R2 = 0.90, v2 = −9.8) on a different harness and a different param path, which is
  worth more than the missing 1.2 Myr of post-collapse wall time. WW is now in the Batch 0 grid.

  It also settles the last open question about scope. A new harvester column, `frac_nIFStr_eq_n0`,
  detects cap-binding **without** the Batch-1 diagnostic — the stored `n_IF_Str` came from the
  `min()` iff it sits exactly at `shell_n0` — so it can be read on baseline runs and, crucially,
  separates a genuine cap-slack row from a merely stale-`Pb` row. Measured: **1.0000 in every phase
  of all six configs, including WW.** WW's momentum phase has one row at `P_HII/Pb = 0.787`, which
  looked like the cap finally going slack in the weak-wind collapse — the very regime weak-winds
  predicted it would. It is not: `n_IF_Str/shell_n0 = 1.000000` there, so the cap is bound and the
  offset is `Pb` staleness during fast collapse (`Pb := pRam` moves quickly when v2 = −10).

  **So the identity is total across the tested matrix**: there is no config, phase, or row where
  `P_HII` carries independent physics. Every apparent exception — 1a→1b handoff, collapse — is
  `params['Pb']` moving between the `shell_structure` call that set `shell_n0` and the snapshot
  write. Two consequences: (i) a fix cannot be scoped to "the regime where the cap binds", because
  that is everywhere; (ii) weak-winds' caveat that a weaker wind might leave the cap slack is
  **not** borne out at c = 0.1 — if that transition exists it is further out, and Batch 4 should
  look for it with the diagnostic rather than assume it.

  *Not done:* WW has no `b1` (diagnostic) arm, so it has no blow-up number. It is the most likely
  place for a large `raw/shell_n0` — weak winds, small ΔV, collapse — so **run `b1` WW before
  Batch 4a** and re-check the kill bar against it.

- **2026-08-12 (Batch 4a + b1 WW)** — b1 WW re-run after a container restart killed it; bit-identity
  gate now covers **593 rows across 3 configs**, all exact. Batch 4a (bare cap removal, C2a) run on
  PRB/B3M/F1HI/F1LO from a throwaway worktree so the main tree stayed clean and both arms could run
  concurrently; the cap edit is deliberately uncommitted (4a is a measurement, not a proposal).
  **Result: C2a survives cleanly on all four — zero numerical distress, no fate changes, faster than
  baseline — but breaches the 5% trajectory bar on every config (15.3–28.4%).** The maintainer's
  "i dont know if it breaks things" is answered: it does not break, but it moves the answer.
  Two things changed in this doc as a result: (1) the Batch-4 section gained the result table and an
  explicit **retraction** of the earlier (wrong) prediction that removal would *lower* the drive —
  it raises it, because the cap clamps downward; (2) D2 is promoted from a background question to
  **the crux** — with removal shown safe, the only remaining question is whether the uncapped
  Strömgren pressure is physically trustworthy at these ionized volumes, and no measurement here can
  settle that. Re-ranking: 4b (guard replacement) now matters more than C1 alone, because 4a shows
  the identity is load-bearing at the ~2× level rather than cosmetic; but both still wait on D1/D2.
  Added `harness/compare_trajectories.py` (matched-t, validated against the b0-vs-b1 null at 0.000%).

- **2026-08-12 (D1/D2 answered; the diagnosis moves)** — Maintainer ruled: the momentum
  `P_HII + P_ram` sum is **intended**, conditional on `P_HII` being genuinely its own calculation;
  the transition `max` is a **deliberate** smooth handover as `Pb → 0`; and `P_HII` should be a real
  separate pressure. Three consequences, all recorded above. (1) **C1 is rejected** — replacing the
  momentum `+` with a `max` fixes a thing that is not broken; the defect is the circularity
  `P_ram → Pb → shell_n0 → n_IF_Str → P_HII`, not the sum. (2) New §3b proves, with N = 803 rows over
  8.8 decades, that **the cap is only the last link**: `ΔV ∝ shell_n0^-2.126` and
  `n_IF_Str ∝ ΔV^-1/2` give `n_IF_Str ∝ shell_n0^+1.039` (r = 0.993) *before* the cap applies, and
  Batch 4a's uncapped runs still never put `P_HII` below `Pb`. The intervention point is the shell
  ODE's inner boundary condition at `shell_structure.py:124-126`, not the cap at `:251`.
  (3) **Batch 4b is deprioritised** — replacing the guard cannot decouple anything, because the
  guard is not what couples. Batch 5 (C3) is promoted to next, and its first stage is an *offline*
  screen of C3a/C3b against committed snapshots, since both are closed-form in already-stored
  quantities and the likely failure mode is magnitude rather than stability. C3a measured offline on
  B3M momentum rows: 235 → 49 cm⁻³, P/k 2.4e6 → 4.9e5 K cm⁻³, decoupled, but ≈5–7× `P_ram`.
