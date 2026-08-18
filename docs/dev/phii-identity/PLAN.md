# PLAN — fixing the P_HII identity (branch `bugfix/phii-pt1`, merged to `main` 2026-08-14)

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

**Status (2026-08-14):** 🟡 **C3c IS IN PRODUCTION — this doc is no longer describing a candidate.**
`c43a50e` (PR #738, merged `186cc5a`) replaced the capped-Strömgren `P_HII` at all six call sites in
the four phase runners; the helper is `get_bubbleParams.get_phii_c3c` @ `c43a50e`. Batches 0, 1, 3,
4a and 5-stages-1/1b/2/3 done; **D1-D4 answered**. What is still open is the **momentum branch** (a
modelling call, §Batch-5-stage-3 below), not the landing. An independent adversarial audit
(2026-08-13) corrected 2 critical + 11 major items — see §9, and do not quote any figure from an
unmarked earlier revision of this file.

⚠️ **Two consequences of landing, both foreseen here and neither finished:**
(1) **D-ramp is now fixed in production** — the third consequence §3c predicted for C3c, defect
defined at §3 item 3. The energy-phase drive is the ramped `Pb` alone, so the phase-1a-exit goldens
moved by −1.1%: `test_run_smoke.py`
`R2` 0.25955976 → 0.25672223 and `test_phase_boundary.py` `cool_beta` 0.888197 → 0.878395.
`test_betadelta_hybr_stress.py` carries the same `(0.888197, -0.046294)` pair and is red too —
measured 2026-08-14, `cool_beta` = 0.87839528 at t=0.00350 (`TRINITY_STRESS_N=1 pytest
test/test_betadelta_hybr_stress.py -m stress`, 10 min) — but it is stress-marked, so it does not
show in a default `pytest` or in CI.
D4 granted re-baselining authority for `test_phase_boundary.py`, `test_betadelta_hybr_stress.py` and
the `test_scheme_screen.py` fixtures conditional on G3.4's before/after table — **`test_run_smoke.py`
is not on that list and needs its own sign-off.**
**Status update (2026-08-18):** 🔴 **SHIP-HOLD on all further P_HII work.** The self-consistency
audit (§6b below) found four driving-branch seams — photon double-spend, boundary-pressure mismatch,
a mass double-book measured at `M_cav/M_shell` → 0.56, and thin-shell strain. The maintainer's
ruling: these look like real problems that block shipping and need investigation, **and the audit
itself must not be assumed correct** — every seam gets an adversarial re-verification before
anything is built on it. Registered as **Batch 11**. On hold behind it: the `phii_scheme` key
(D5: C3c-switch vs C3a-raw), any default change, and quoting the driving-branch
`P_HII/P_ram` ≈ 6 momentum numbers as physics. The confined branch (energy/implicit) is unaffected
— the audit found it exactly consistent, and Batches 7/8 stand.

**B11.0 DONE (2026-08-18) — the hold stands, and it got firmer, not softer.** The adversarial
re-verification could not kill any seam. **A CONFIRMED** (no cavity-absorption factor exists at any
`Qi` consumer; `f_abs` = 1.0000 on 29/33 driving rows ⇒ the claimed photon budget is ≈`2·Qi`).
**B REVISED** — the mismatch is real and sized (`P_HII/Pb` median 6.16 momentum / 4.62 transition)
but its *direction* was wrong: the loop back into `P_C3a` is exactly zero on 88% of driving rows and
**upward** on the rest, so B is not an upper-bound mechanism and §6b's "every seam pushes the same
way" is corrected. **C CONFIRMED to 4 s.f. and understated** — 0.0952 → **0.5638**, reproduced by
three independent routes (two agreeing to 1e-12), units cleared by `units-reviewer`; and the shell
already carries **100.0000%** of the gas the run has, with winds supplying **54.8 Msun** total, so
`M_cav` has no source and `(M_cav + M_shell)/M_avail` = **1.5638**. **D CONFIRMED** (`dR/R2`
0.658–1.308). Evidence: `data/b11_mass_ledger.csv`, `harness/mass_ledger_check.py`.
Two consequences beyond the seams: B11.B is re-scoped (it would have "fixed" the standard side), and
**Geen et al.'s two-equation closure has none of the four seams by construction**, giving D5 an
external reference model instead of a from-scratch design (B11.G). Three side-findings (S1–S4)
correct committed Batch 9/10 numbers; see B11.0 RESULT.

(2) **`switchon-successor/` measured `dt_switchon` in the regime C3c has now removed.** Every batch
there ran with `P_HII == Pb` un-ramped winning the `max`, so the ramp was inert in the momentum
equation; it now throttles `vd`. Its algebraic results (D1, D4) survive; its ablation and Weaver-N1
figures do not. Flagged in `docs/dev/DOC_STATUS.md`; that workstream owns the re-run.

- **The identity is real and universal**: `P_HII` == the confining pressure to ≤2.9e-16, and the cap
  binds on **100% of rows in every phase** of every config (6 configs, 4 decades of `nCore`).
- **The cap is not the coupling** (§3b). Removing it (Batch 4a) leaves `P_HII` still tracking `Pb`
  — per-config slope **+0.996…+1.096**, and `P_HII < Pb` on **zero** rows. The coupling runs through
  the ionised volume, because `shell_n0 = Pb/(kT)·μ` is the shell ODE's inner boundary condition.
  **The intervention point is `shell_structure.py:124-126`, not the cap at `:253`.**
- **The double-count is measured, both ways.** Momentum drives on exactly `2·P_ram`; transition
  overshoots by 1.82× median. Batch 3 then measured what removing it costs: **≤4.0% ΔR2**, no fate
  changes, on weak-wind/two-mass/two-radius configs. C1 is safe but wrong-target (it deletes the
  photoionised channel instead of decoupling it), so it is superseded by C3 and kept as the price tag.
- **Batch 5 stage 1 (offline, no solver run): C3b ⛔ rejected** — `n = n_cloud(R2)` has no `Qi`
  dependence, failing the pre-registered wind-only limit structurally. C3a passed but asserted
  cavity-filling everywhere.
- **Batch 5 stage 1b: C3c designed (§3c) and screened — it is the surviving candidate.** The design
  pass proves a confined skin has no independent density, so C3c is a **regime switch**: transmit
  while `P_C3a ≤ P_conf`, drive at `P_C3a` above. On the stock trajectories it leaves the implicit
  phase *exactly* untouched, fixes D-ramp as a side effect, puts the handover crossover inside the
  transition phase in all four configs that reach it, and raises the momentum drive to 2.4–4.3×
  stock. The coevolution crossover D2 asked about **does** appear — vs the confining pressure, in
  transition — not vs `P_ram` within momentum.
- **Batch 5 stage 2 DONE: the C3c arm runs clean.** 5/5 configs, **zero numerical distress**, **no
  fate changes**, and the pre-registered null passed exactly (`P_HII`=0 on 0/330 implicit rows). All
  configs are OVER-BAR at 12.8–20.5% ΔR2, which was pre-registered as expected rather than as a
  failure. WW collapses 16% earlier but still collapses. The `t_cross` kink did not trouble the
  integrator, so §3c.1's event-detection remedy is registered but unneeded so far.
- **Batch 5 stage 3 DONE: the wind ladder splits the Lancaster verdict.** The first ladder (on
  `simple_cluster`) was **void** — three rungs never reach the transition phase, so their
  `t_cross = never` says nothing about winds. Re-run on B3M, all four rungs valid. **Resolved in
  transition:** the confined fraction grows 8.8% → 38.8% across two decades of wind and
  `ratio@entry` follows `Lw^−0.743` against a pre-registered −0.74 (per-rung error 2–7%), so C3c
  reproduces wind-dominated regimes with a Weaver-derived exponent. **Open in momentum:** all four
  rungs stay 100% HII-dominated; `P_C3a/P_ram` falls only as `Lw^−0.33`, so inversion needs
  `Lw ≈ 260`. Crucially this is **not** an O(1) normalisation error — the same normalisation
  predicts transition to within 7%. It is the **exponent**, i.e. the `R2^−3/2` cavity-Strömgren
  geometry, not the prefactor.
- **D3 and D4 ✅ ANSWERED 2026-08-13** (§7): WW's 16%-earlier collapse is accepted as an explained
  timing change; golden re-baselining is authorised subject to G3.4's before/after table.
- **Landed 2026-08-14 (`c43a50e`), ahead of Batch 6's full-12 matrix.** The one physics question
  left open is the momentum result above, which is a modelling call (accept photoionisation-dominated
  momentum as the prediction, or revisit the cavity geometry), not something another ladder settles.
- **Next (re-ordered 2026-08-18, updated after B11.0):** (a) **Batch 11** — B11.0 ✅ done (all four
  seams survive; B revised); next is **B11.A** (photon-conserving fixed point), **B11.B** (re-scoped:
  measure the inconsistency, don't "fix" the shell) and **B11.C** (mass-ledger consequence — note
  its sub-question (i) is now partly answered: there is no supply, so the supply-limited limb is the
  only available one). Everything else P_HII still waits on it; (b) the G3.4 before/after table +
  golden re-baseline; (c) hand the `switchon-successor` re-run to that workstream; (d) low-priority
  but cheap: **B11.F** (re-fit Batch 9/10 on the exact layer volume) and **B11.G** (score against
  Geen et al.'s closure).

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
(`shell_structure.py:124-126`), `n_IF_Str` is capped at it (`:253`), and `P_HII` converts back
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
   un-ramped R1. Inside that window the two differ by up to **3.31×** (WW energy), so the `P_HII`
   channel reintroduces exactly the pressure the ramp exists to suppress. Call this **D-ramp**.

   ⚠️ **Retracted 2026-08-13, in place.** An earlier revision of this bullet ended: *"This is the
   single biggest risk to any cap fix: removing the cap drops early driving pressure by up to 3×,
   which will look like the fix breaking the code."* **That is backwards.** The cap clamps the
   Strömgren density *downward*, so removing it raises `P_HII` above `Pb` and the drive goes **up**
   — Batch 4a measured median `P_drive/Pb` rising and ΔR2 growing, never falling. The wrong
   sentence survived here, inside the block a reader is told to treat as the corrections, for a full
   day after being retracted at §6; it is struck now. Flagged by the independent audit.

## 2. Maintainer input on record

**2026-08-12 (this session):** the cap's origin is numerical, not physical — *"Originally i had
the PHII cap because at small volume it'd give very high n_str_if and that would give very high
PHII, and i dont know if that breaks things."* At small ionized volume ΔV → 0 the Strömgren
balance `n_IF_Str = sqrt(3(1−f_esc)Qi / (4π χ_e αB ΔV))` (`shell_structure.py:247-250`) diverges;
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
  ⚠️ **An earlier revision of this line marked C1 ⛔ REJECTED on the strength of this ruling alone.
  That was wrong and is retracted (2026-08-12): a verdict was declared from stated intent with no run
  behind it, which violates this workstream's own bar. C1's status is reset to ⬜ pending **Batch 3**,
  which must measure it across configs — weak-wind, high-sfe, low-sfe, both density extremes — before
  any verdict. The intent ruling is an *input* to that verdict, not a substitute for it.**
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
| **C2a** | *bare cap removal* (`shell_structure.py:253` deleted) | ALL phases (1a/1b via `max(Pb, P_HII_raw)`, which un-absorbs whenever raw > `shell_n0`… i.e. exactly where the cap used to bind) | **high** — the ΔV→0 divergence the cap was built for; interacts with the per-segment freeze ratchet (phase1a-init Extra finding #1) which is *catastrophic* at compact scale | tiny diff, expensive validation |
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

**The real coupling runs through the ionised volume.** Regressions on the pre-cap diagnostic over
the stock (b1) trajectories:

Artifact: `data/b3b_coupling_regression.csv` (`harness/coupling_regression.py`), over the **four
complete b1 runs** (B3M, PRB, WW, B1M; 788 rows, 8.77 dex of `shell_n0`).

| relation | pooled | per-config spread | meaning |
|---|---|---|---|
| `shell_n0` vs `Pb` | exact (`shell_structure.py:124-126`) | — | the inner BC *is* `Pb/(k T)·μ` |
| `log ΔV` vs `log shell_n0` | slope **−2.348**, r = −0.940 | −2.02 … −2.72 (r ≤ −0.985) | denser shell ⇒ thinner skin |
| `n_IF_Str ∝ ΔV^(−1/2)` | by construction (`:247-250`) | — | — |
| ⇒ `log n_IF_Str_raw` vs `log shell_n0` | slope **+1.036**, r = **0.994** | **+0.996 … +1.096** | `P_HII` tracks `Pb` linearly |
| uncapped `P_HII`/`Pb` | 1.031 … **7.786** | frac < 1 = **0.0000** everywhere | `Pb` is never the larger one |

Read the **per-config spread as the result**, not the pooled exponent: the pooled fit is a
between-config average whose r is *worse* than any individual run, and the closure
"−2.348 × (−½) ≈ +1.04" is forced by construction, so it corroborates the arithmetic rather than
testing it independently. The load-bearing fact is that `n_IF_Str_raw` tracks `shell_n0` with an
exponent within 10% of 1 **in every config**, and never falls below it.

⚠️ **Corrected 2026-08-13 (independent audit).** An earlier revision quoted −2.126 / +1.039 from a
pool of N=803 that included two runs killed mid-flight, one duplicating ~110 rows of its own
completed re-run (C-7), and cited no artifact at all. Conclusion unchanged; digits were not
reproducible.

The chain closes:

```
Pb ──> shell_n0 ──> [shell ODE inner BC] ──> R_IF ──> ΔV ∝ shell_n0^-2.13
                                                        │
                          n_IF_Str ∝ ΔV^-1/2 ∝ shell_n0^+1.06 ≈ Pb  <──┘
```

**So the intervention point is `shell_structure.py:124-126`, not `:253`.** The Strömgren balance is
being evaluated over a volume whose size is set by `Pb`, so it cannot report anything but `Pb`. Any
decoupling must break the ΔV path; removing the cap alone provably does not.

### C3 candidates, re-derived against this diagnosis

- **Why C3a carries no cloud term at all (maintainer question, answered 2026-08-18).** The absence
  is not an omission — it is a property of the classical solution itself. Spitzer's interior density
  is `n_i = n_0 (R_St/R)^{3/2}`, and substituting `R_St = (3Qi/(4π χ_e α_B n_0²))^{1/3}` makes the
  ambient density **cancel identically**: `n_i = sqrt(3Qi/(4π χ_e α_B R³))` — exactly C3a (verified
  numerically over 3 decades of `n_0`; G8.1 pins the boundary case `n_i(R_St) = n_0`). The cloud's
  entire influence on the *classical* interior is channelled through where the front is, and C3a
  inherits that structure: in trinity the cloud enters through **`R2(t)`** (the ODEs it feeds —
  swept-up `shell_mass`, gravity, the density profile all shape the trajectory C3a is evaluated on)
  and through **`f_abs`** (the shell solve over real swept-up cloud material). The closure is
  legitimate because ionisation equilibrium is effectively instantaneous: at the measured B3M cavity
  densities, `t_rec = 1/(χ_e n α_B)` is 1.7e-8–6.7e-6 Myr in energy and 4.7e-4–3.5e-3 Myr in
  momentum, i.e. 2–5 decades below the simulation time everywhere. Where a cloud term *genuinely*
  belongs and is idealised away: (a) **mass supply** — C3a assumes photoevaporation off the shell
  can always fill the cavity to the Strömgren density (the "Strömgren-filled" model-structure
  question; a supply-limited region is density-bounded and pushes with less); (b) **dust**, a cloud
  property — G9.4 measured it consuming 61–75% of `Qi_abs`, lowering the real recombination-
  equivalent density up to 3.2× below the dust-free balance. C3b sat at the opposite pole — all
  cloud term, no `Qi` — and was rejected for exactly that (no wind-only limit).
- **C3a — Strömgren over the cavity.** `n_HII = sqrt(3 Q_i,abs / (4π χ_e α_B R2³))`: depends on
  `Qi` and `R2` only, **zero** dependence on `Pb`, `P_ram` or `shell_n0`. Measured offline on B3M's
  momentum rows: n = **235 → 47 cm⁻³** over R2 = 6.6 → 19 pc, i.e.
  **P/k = 5.2e6 → 1.0e6 K cm⁻³** once the `mu_convert/mu_ion_shell` = 2.2 factor that the code's own
  `P_HII` applies is included — physically reasonable H II magnitudes.
  ⚠️ *Corrected 2026-08-13 (audit): an earlier revision quoted `n·T` = 2.4e6 → 4.9e5 K cm⁻³, omitting
  that 2.2 factor, while quoting the companion "5–7× `P_ram`" ratio that does include it — the two
  were mutually inconsistent, and the "physically reasonable" verdict rested on the low one. The
  density also shifts 49 → 47 cm⁻³ using absorbed `(1−f_esc)·Qi` as the code's balance does, rather
  than total `Qi`. Still has no committed harness or CSV — see the §5 note below.* Scaling `P_HII ∝ R2^-3/2` vs `P_ram ∝ R2^-2` means the two **cross
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

## 3c. C3c designed (2026-08-13): the confined skin has no independent density — so C3c is a regime switch

D2's brief for C3c was "keep the skin geometry, decouple its `Pb`-derived inner boundary condition."
The design pass shows that brief, taken literally, is **ill-posed** — and what survives of it is a
two-branch formulation that reuses C3a as one branch. The elimination argument, kept because it is
the actual content of the design:

1. **Independent-thickness skins are C3a × O(1), on the wrong side.** Any Strömgren balance over a
   skin of decoupled thickness ΔR gives `n ∝ ΔV^(−1/2)`; a skin's ΔV is *smaller* than the cavity's,
   so its pressure is **higher** than C3a's — e.g. ΔR = 0.1·R2 gives ≈1.8× C3a. Since C3a is already
   3.5–7.6× `P_ram` in momentum, every member of this family makes the overshoot worse. In ionization
   equilibrium at fixed absorbed `Qi`, **the cavity-filling configuration (C3a) is the *minimum*
   pressure**; confinement can only raise it.
2. **Jump-condition closures re-couple.** Setting the ionized density from the I-front jump condition
   ties it to the neutral shell's density — which is pressure-confined by the drive. Circular again.
3. **Mass closures re-couple.** Setting it from the ionized shell mass imports the shell-structure
   integration, whose inner BC is `Pb`. Circular again.
4. **The remaining closure is pressure equilibrium with the confinement — which is the current code.**
   A confined skin in equilibrium *has no independent density*: its density IS the confining pressure
   restated. That is not a bug in the implementation; it is what the cap has been measuring all along.

**So the physical content is a regime switch, not a new density formula:**

```
P_free = P_C3a = (μc/μi)·kT·sqrt(3(1−f_esc)Qi / (4π χe αB R2³))     (the relaxed equilibrium)

if P_free ≤ P_conf :  wind/bubble CONFINES the ionized gas → it is a thin skin at P_conf,
                      transmits the confinement, contributes NOTHING independent (F_HII_indep = 0)
if P_free > P_conf :  confinement cannot hold → ionized gas fills its own volume and DRIVES
                      at P_C3a, which self-regulates (∝ R2^(−3/2)) as the shell moves out
```

Per-phase drive under C3c, honouring D1's rulings (momentum = sum; transition `max` = deliberate
handover):

| phase | stock | C3c |
|---|---|---|
| energy / implicit | `max(Pb_eff, P_HII≡Pb)` | `max(Pb_eff, P_C3a)` |
| transition | `max(Pb, P_HII + P_ram)` | `max(Pb, P_C3a + P_ram)` |
| momentum | `P_HII + P_ram = 2·P_ram` | `P_C3a + P_ram` on the driving branch; **`P_ram` alone on the confined branch** |

⚠️ *Clarified 2026-08-13, prompted by a maintainer question:* the branch rule is primary **in every
phase**, momentum included. If `P_C3a ≤ P_ram` there (strong winds, faded `Qi`), the wind confines
the ionized gas and the skin contributes nothing independent — the drive is `P_ram` alone, not the
sum. The table's earlier unconditional `P_C3a + P_ram` was written against the current screen data,
where momentum is HII-dominated on 100% of rows, so the confined-momentum corner never arose; the
regime map (stage 3) is designed to reach it.

Three consequences fall out **without further design**, all screenable offline:

- **D-ramp is fixed as a side effect.** In the energy phase `P_C3a ≪ Pb`, so the `max` selects the
  ODE's own (ramped) bubble pressure — there is no longer a fictional `P_HII` carrying the un-ramped
  pressure past `dt_switchon`.
- **The transition `max` finally binds physically.** The screen shows `P_C3a/Pb` crossing 1 *inside
  the transition phase* in **every config that reaches transition** (B3M 0.12→5.1, WW 0.49→4.3,
  B1M 0.36→6.8, B2M 0.17→7.4). The handover the maintainer intends the `max` to represent is exactly
  the moment the dying bubble pressure falls under the HII-region pressure. ⚠️ This also corrects the
  Batch-5 stage-1 statement that "the coevolution crossover does not appear": it does not appear vs
  `P_ram` *within* momentum, but the physically relevant crossover — vs the confining pressure —
  appears in transition, in all four configs that get there.
- **The cavity-filling assumption is only invoked where it is plausible.** C3a asserted ionized gas
  fills the cavity *always*; C3c asserts it only on the branch where the wind is too weak to keep the
  cavity evacuated — which is precisely when filling is credible.

### §3c.1 Continuity across the regime switch and the phase seams (added 2026-08-13)

Raised by the maintainer: does the formulation stay smooth at the regime switch and across the
energy/transition/momentum seams? Measured offline on the same five runs — artifact
`data/b5_c3c_seams.csv` (adjacent-snapshot drive ratios; each is discontinuity *plus* one segment of
real evolution, so an upper bound — but stock and C3c use the same row pairs, so the comparison is
fair):

| seam | stock jump (after/before) | C3c jump | reading |
|---|---|---|---|
| energy → implicit | 0.81–0.84 | **0.89–0.92** | C3c smoother — no fictional `P_HII` discontinuity riding the ramp end |
| implicit → transition | 0.59–0.99 | 0.53–0.96 | comparable; see the `P_ram` note below |
| **regime switch** (in transition) | 0.75–0.83 *(same rows)* | **0.86–0.99** | the switch is nearly invisible: `max` is C0 — at the crossover both branches are *equal*, so the drive is continuous and only its derivative kinks |
| transition → momentum | 0.993–0.996 | **0.995–0.999** | continuous **by construction**: `P_C3a` depends only on `Qi, R2` and does not know the phase label, unlike stock's `P_HII`, which is redefined (`Pb`-slaved → `P_ram`-slaved) at this seam |

Three structural points, beyond the numbers:

1. **Within a phase, C3c is C0-continuous with derivative kinks only** — that is what `max()` gives.
   At `t_cross` the two branches cross at equal value; there is no drive jump to integrate over. The
   stiffness question at the kink is the same class the code already handles at every `max` it
   contains today.
2. **The `P_ram` switch-on at implicit → transition is a stock discontinuity that C3c *mediates*.**
   Stock's transition drive is effectively `Pb + P_ram` from the first transition step — `P_ram`
   turns on instantaneously. Under C3c the `max` clips it: the drive stays `Pb` (continuous with
   implicit) until the physical crossover, and `P_ram` enters only as part of the HII-branch
   takeover. This is why C3c's raw ratio at this seam is *slightly lower* than stock's in WW
   (0.526 vs 0.586): stock's instantaneous `+P_ram` partially masks the genuine decline of `Pb`
   across the gap. The smoother-looking stock number is the artifact; the discontinuity is stock's.
3. **If the stage-2 arm shows the integrator stumbling at `t_cross`, the pre-registered remedy is
   event detection, not smoothing constants.** `trinity/phase_general/phase_events.py` already
   provides the factory machinery (`make_cloud_boundary_event` precedent) to let the solver land on
   the crossover exactly instead of stepping over it. A smooth-max (softmax / p-norm) blend is the
   fallback, but it introduces a width parameter — a new magic number, which this codebase is
   actively purging — so it requires a maintainer ruling before use. The `dt_switchon` ramp is the
   in-repo precedent for time-smoothing if a ramped variant is ever preferred.

**Caveats, stated up front:** the C3a branch carries an O(1) normalization ambiguity (uniform sphere
vs the R1..R2 shell — small since R1³ ≪ R2³; photoevaporative-flow closures land on the same
√(Qi/R2³) scaling with different O(1) factors). And in the confined branch, "contributes nothing
independent" is a *model choice* consistent with D1's "decoupled `P_HII`" instruction — the skin
still transmits `P_conf`, it just stops being double-counted as a separate entry.

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
**pre-cap** value as `n_IF_Str_raw` (pattern: `n_IF_ODE` at `:226` — a raw value already kept for
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
  **identical to C1 wherever the cap binds**, which Batch 1 shows is everywhere. ⚠️ *Rewritten
  2026-08-13 (audit): the original rationale localised the difference to "cap-slack windows (early
  1a, where `P_HII = 3·Pb` genuinely drives)" — that premise was retracted by Batch 1 and would have
  carried a known-false assumption into this experiment's design. The real difference is **D-ramp**:
  inside the `dt_switchon` window `P_HII` carries the un-ramped bubble pressure, so an
  `include_PHII=False` arm removes that too and over-states what a momentum-only fix would do.*
- **Confound, documented up front:** the off arm also removes the *legitimate* early-1a HII
  driving; interpret early-time ΔR2 as envelope, not as C1 prediction. B1's cap map says exactly
  where the confound lives.
- **PASS bars:** screen completes on core-6; ledger records ΔR2(t), Δfate, Δt_end per config;
  no bar on the *size* of Δ (this batch measures, it does not judge).
- Artifacts: `data/b2_bracket_ledger.csv`.

### Batch 3 — C1: transmit-don't-add — Status: ✅ **MEASURED 2026-08-13**

**Result.** Arm = `harness/b3_c1_momentum_max.patch` (momentum sites only — `run_momentum_phase.py:265,445`;
§3's "5-site" wording predates D1's ruling that the transition `max` is deliberate, so the transition
sites were deliberately left alone). Baselines are the matching b1 arms; matched-`t` ledger in
`data/b3_c1_ledger.csv`.

| config | what it spans | momentum rows | ΔR2 max | fate |
|---|---|---|---|---|
| **B1M** | 5e4 M☉, r20 — **never reaches momentum** | 0 / 195 | **0.000%** | unchanged |
| B2M | 1e5 M☉, r10 | 26 / 225 (11.6%) | 1.243% | unchanged |
| B3M | 1e5 M☉, r5 | 34 / 231 (14.7%) | 4.003% | unchanged |
| WW | **weak winds** (`FB_thermCoeffWind` 0.1) | 27 / 178 (15.2%) | 1.291% | unchanged |

- **The control did its job.** B1M was pre-registered as a falsifiable check: C1 touches only the
  momentum phase, B1M never enters it, so C1 *must* be inert — and ΔR2 is 0.000% at matched `t`.
  (Its `dictionary.jsonl` is not bit-identical, but every differing key traces to `Lmech_SN`, which
  is `Lmech_total − Lmech_W` and therefore exactly zero before SN onset at 3.6 Myr; the stored ~1e-18
  is a cancellation remnant ~1e-26 *relative*, and it seeds last-bit integrator drift reaching only
  2.9e-14 in R2. That measurement is what set `NOISE_FLOOR` in `compare_bitidentical.py`.)
- **Halving the momentum drive costs ≤4% in final radius** on these configs, because momentum is
  only 12–15% of the run. Weak winds is *not* the worst case (1.29%); the densest bench is (4.00%).
- **Nothing broke** — no fate changes, no distress.

**Verdict: C1 is viable but aimed at the wrong target.** Two independent reasons, one measured and
one from intent: (1) with `P_HII ≡ P_ram` in the momentum phase, `max(P_HII, P_ram)` is just `P_ram`
— C1 removes the double-count by *deleting* the photoionised channel, which is the opposite of D2's
"`P_HII` should be a real, separate pressure"; (2) D1 rules the sum intended. C1 is therefore
**superseded by C3**, and its value is as the measured price tag on the double-count: ≤4% ΔR2, which
is the number to beat when judging whether a C3 formulation is worth its complexity.
⚠️ Note this verdict now rests on runs, not on the intent ruling alone — the earlier
intent-only rejection was retracted before these runs existed.

### Batch 3 (original pre-registration) — Status: superseded by the result above
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
  - **Arm:** the cap line (`shell_structure.py:253`) commented out in a throwaway git worktree, so
    the main tree stays clean and the b1 WW run could proceed concurrently. Runs land under
    the **worktree's** `outputs/phii/b4a__088a8d6_dirty/` — never inside this repo. The edit is
    deliberately NOT committed to the branch: 4a is a measurement, not a proposed change.
    ⚠️ **The 4a raw run dirs no longer exist** (2026-08-13, audit): the worktree was removed with
    `git worktree remove --force` once the ledgers were written, taking every `dictionary.jsonl`
    with it. Only `data/b4a_ledger.csv` and `data/b4a_identity_grid.csv` survive. **Any 4a claim not
    in those two CSVs is now unverifiable** — specifically "zero distress lines" (read from the
    deleted logs), "`P_HII/Pb` never below 1 on any row", and "median `P_drive/Pb` 1.83 (PRB)"
    (a pooled-over-phases figure matching no cell in the grid, whose PRB medians are 1.949 and
    1.041). Re-run from `harness/b4a_cap_removal.patch` into a tracked path before quoting them.
- **4b (guard replacement, C2b):** pick ONE replacement guard by B1's data (the one that binds
  only in the blow-up regime and nowhere else), screen it identically.
- **PASS bars:** run survival on all arms; identity broken (relΔ `P_HII` vs `Pb` becomes O(1)
  where the old cap bound); ΔR2/fate ledger complete; PRB terminates.
- Artifacts: `data/b4a_ledger.csv`, `data/b4a_identity_grid.csv`.

**4a result.** Ran **4 of the pre-registered core-6** — SC and WW were never run, and WW is the
config this plan itself flags as the likeliest place for a large blow-up (weak winds, small ΔV,
collapse), which the corrected capmap confirms at 7.79×. So "4/4" below means four of six, not
complete coverage. Of the bars that were exercised, every one is met except the trajectory bar,
which is breached everywhere:

| config | wall_s | fate base → 4a | ΔR2 max | at t | ΔR2 end | distress |
|---|---|---|---|---|---|---|
| PRB | 764 | stopping_time → stopping_time | 28.4% | 1.3e-07 | **0.95%** | none |
| B3M | 625 | stopping_time → stopping_time | 27.4% | 9.0e-06 | 13.6% | none |
| F1LO | 762 | stopping_time → stopping_time | 25.9% | 1.4e-04 | 14.4% | none |
| F1HI | 492 | shell_collapsed → shell_collapsed | 15.3% | 3.2e-06 | 6.2% | none |

- **Survival: unambiguous.** No stalls, no overflow, no monotonic-guard rejection, no convergence
  warnings, and 4a's wall times are comparable to baseline (492–764 s vs 682–832 s; PRB is 7% *slower*, 764 vs 713 s). The ΔV→0 blow-up the cap was built
  for does not materialise in any regime tested — including PRB, the compact probe chosen precisely
  because it is the most likely place for it.
- **Identity destroyed, as intended.** `frac_PHII_eq_Pb` = 0.0000 in every phase of every config.
  `P_HII` again depends on `Qi`, `f_esc` and the ionized volume.
- **But it is a real physics change, not a cleanup.** Uncapped `P_HII` exceeds `Pb` on 100% of rows
  (up to 7.79×; the 3.36 quoted earlier was PRB's `blowup_max`, not the matrix max), so it now *wins* `max(Pb, P_HII)` everywhere and the median `P_drive/Pb` rises
  1.0000 → 1.83 (PRB). Every ΔR2 maximum sits inside the `dt_switchon = 1e-3` Myr window: the cap
  was doing its heaviest work exactly where the R1 ramp is active.
- ⚠️ **Retraction (2026-08-12).** An earlier reading of the §1 correction predicted cap removal would
  *lower* early driving pressure (by removing `P_HII`'s un-ramped-pressure smuggling). That is
  backwards and the data says so: the cap clamps the Strömgren density *down*, so removing it raises
  `P_HII` above `Pb` and the drive goes **up**. Recorded so the wrong prediction is not re-derived.
- **Open, and it is now the crux:** the larger uncapped `P_HII` is only trustworthy if the Strömgren
  balance is trustworthy at these ionized volumes. Nothing measured here settles that — it is D2.

### Batch 5 — C3: the advanced method — Status: 🟡 **stage 1 (offline screen) DONE — C3b rejected, C3a advances**

**Stage-1 screen (2026-08-13).** `harness/c3_offline_screen.py` → `data/b5_c3_screen.csv`, over the
five complete b1 runs (B3M, PRB, WW, B1M, B2M). **No solver was run**: both candidates are
closed-form in quantities already in the snapshots, so each is evaluated *on the stock trajectory*.
That answers "what would this pressure have been", **not** "what would the run have done" — a
candidate that survives still needs an arm.

| test | stock | uncapped | **C3a** cavity | **C3b** ambient |
|---|---|---|---|---|
| slope of `log P` vs `log Pb` | **1.0000 / r 1.0000** in every row | 0.37 … 1.19 | 0.02 … 1.15 | ≈ 0 … 0.56 |
| depends on `Qi`? | no (cap erases it) | yes | **yes** | **NO** |
| `P/P_ram`, momentum | 1.000 | 2.1 … 9.9 | 3.5 … 7.6 | 0.02 … 94 |
| crosses `P_ram`? | never | never | never | in transition (B1M, B2M) |
| ionised `n` [cm⁻³], momentum | 2.9 … 2273 | 6.7 … 1.8e4 | 19 … 8055 | 1 (ISM) … 1e5 |

⚠️ **Read the slope test carefully.** Stock scores *exactly* 1.0000/1.0000 because it **is** `Pb`.
C3a's 0.7–1.1 in several phases is **not** causal coupling — C3a never reads `Pb`; both quantities
simply decline together along a trajectory. Shared time-dependence is not dependence, and this
column cannot separate them. The discriminator that does is the `Qi` row.

**C3b — REJECTED.** It fails the acceptance floor this plan pre-registered before any C3 was
designed ("reproduces the wind-only limit: matches C0 when `Qi → 0`"). `n = n_cloud(R2)` has **no
`Qi` dependence at all**, so switching the ionizing source off entirely leaves its `P_HII`
unchanged. That is structural, not a tuning problem. Two lesser faults confirm it: the value is the
*neutral* gas ahead of the shell rather than the ionised gas pushing it, and it steps
discontinuously from `nCore` to `nISM` at `rCloud`, collapsing to ~1 cm⁻³ exactly where the momentum
phase lives (B3M, B2M) — while for WW, which collapses inside the cloud, it instead reaches 94×
`P_ram`. A driver that swings four decades on a geometric boundary is not a pressure law. ⛔

**C3a — PASSES stage 1, advances to an arm.** It is causally decoupled (`Qi` and `R2` only), it has
the correct `Qi → 0` limit by construction, and its ionised densities are physically sensible
(19–8055 cm⁻³ in momentum, i.e. P/k ≈ 4e5–2e8 K cm⁻³). **But it is not a small change**: it sits
uniformly **3.5–7.6× above `P_ram`** in the momentum phase of all five configs and never crosses,
so it predicts a *photoionisation-dominated* momentum phase everywhere. Whether that is right is a
physics call, not something this screen can settle — but note it is a *falsifiable* prediction, and
the coevolution crossover D2 hoped for does **not** appear within these configs' `R2` range
(`P_C3a ∝ R2^-3/2` vs `P_ram ∝ R2^-2`, so the crossover sits at smaller `R2` than the momentum phase
ever reaches).

**Stage 1b (2026-08-13) — C3c designed (§3c) and screened jointly.** Artifact:
`data/b5_c3c_regime.csv` (same builder, same five runs, still no solver run). The regime structure
comes out exactly as §3c predicts, in every config:

| phase | frac HII-dominated | C3c drive / stock drive (min..med..max) | reading |
|---|---|---|---|
| energy | **0.0000** (all 5) | 0.30 .. 0.70–0.94 .. 1 | confined branch; the <1 ratios ARE the D-ramp fix — C3c honours the ramped pressure, stock's fictional `P_HII` did not (1/3.3 ≈ 0.30) |
| implicit | **0.0000** (all 5) | **1 .. 1 .. 1 exactly** | provable no-op — the falsifiable control, passed on every row |
| transition | 0.76–0.90 | 0.84–0.94 .. 2.5–3.7 .. 3.0–4.2 | the handover: early transition C3c *removes* the 1.82× double-count (ratio < 1), then the HII branch takes over as `Pb` dies |
| momentum | **1.0000** (all 4 that reach it) | 2.4 .. 2.7–4.2 .. 4.3 | HII-dominated everywhere; drive = `P_C3a + P_ram` = 2.4–4.3× stock's `2·P_ram` |

`t_cross` lands **inside the transition phase in all four configs that reach it** (B3M 0.301, WW
0.163, B1M 0.666, B2M 0.449 Myr); PRB never leaves implicit and never crosses — also as predicted.

**Verdict: C3c supersedes bare C3a as the candidate.** Same decoupled physics on the driving branch,
but (i) the cavity-filling assumption is invoked only where the wind is too weak to prevent it,
(ii) the implicit phase is exactly untouched, (iii) D-ramp is fixed as a side effect rather than as
separate work, and (iv) the transition `max` becomes the physically binding handover the maintainer
intends. The cost is unchanged from C3a where it matters: the momentum drive rises 2.4–4.3× over
stock, and the early-energy drive drops up to 3.3× (the ramp finally biting). **Both are large,
real behavioural changes — the stage-2 arm decides whether fates survive them.**

**External anchor — Lancaster's coevolution model (maintainer, 2026-08-13).** The maintainer notes
that Lachlan Lancaster's wind/HII coevolution work finds **wind-dominated regimes** exist. That is a
constraint on how C3c's stage-1 screen should be read, and it cuts two ways:

- It **supports** the C3c *structure* over both stock and bare C3a: an ordering that can go either
  way is exactly what a coevolution model needs, and stock cannot express it at all (its cap makes
  `P_HII ≤ Pb` an identity). C3c's confined branch **is** the wind-dominated regime.
- It is a **tension with C3c's current prediction**, and should be treated as one: the screen finds
  momentum 100% HII-dominated in all five configs. Either (a) these five configs simply do not
  sample the wind-dominated corner — plausible, since none is a strong-wind cloud and all reach
  momentum only after the bubble has failed — or (b) C3a's normalization is too generous by an O(1)
  factor and the true crossover sits inside the momentum phase.

**(a) and (b) are distinguishable, and stage 3 is the discriminator.** If the strong-wind rungs
(`FB_thermCoeffWind` 3/10) push `t_cross` later or out of the run, that is (a) and C3c reproduces
Lancaster's qualitative result. If even strongly-wind-driven clouds come out HII-dominated in
momentum, that is (b) and the C3a normalization needs revisiting (§3c already flags its O(1)
ambiguity: uniform sphere vs the R1..R2 shell, and the photoevaporative-flow closures that land on
the same √(Qi/R2³) scaling with different prefactors). **Recorded before stage 2 runs so the
prediction cannot be retrofitted to whatever comes back.**

**⚠️ The discriminator as worded above is UNSAFE, and the first ladder proved it (2026-08-13).**
"Push `t_cross` out of the run" is not sufficient evidence for (a), because a run can fail to cross
for a reason that has nothing to do with winds. The C3c crossover is **structurally confined to the
transition/momentum phases**: the confinement ratio at transition entry is 0.12–0.49 across every
complete run measured — always below 1 — so energy and implicit are *unconditionally* confined and
`t_entry` is a hard **floor** on `t_cross`. A cloud that never reaches transition cannot cross at any
wind strength.

`simple_cluster` is exactly such a cloud, so the pre-registered ladder was **void**: SC / SW3 / SW10
all terminate at `stop_t` still in the implicit phase (energy 97/105/115, implicit 95/96/96,
`stopping_time`), reporting `t_cross = never` for a reason unrelated to wind. Only WW crosses, and
only because weak winds let the shell *collapse* — collapse is what drags it through transition. Had
this been read at face value it would have been a false confirmation of Lancaster. Ladder re-run on
**B3M**, which spends 42 rows in transition and 34 in momentum (configs `B3MW01/B3MW1/B3MW3/B3MW10`).

**Re-registered discriminator for the B3M ladder (written 2026-08-13 while the runs were still in
flight, results unseen).** All four rungs share one cloud and one SHA; only `FB_thermCoeffWind`
moves. A rung counts as *crossing* only if it reaches transition at all — a rung that terminates in
implicit is **void, not evidence**, and must be reported as such.

- **(a) Lancaster reproduced** — `t_cross` rises monotonically with wind, *and* the lag
  `t_cross − t_entry` rises with wind (not just `t_entry` itself). The lag term is what separates a
  genuine wind-sensitive crossover from the handover simply moving.
- **(b) C3a normalization suspect** — `t_cross` flat or falling with wind, or all four rungs still
  100% HII-dominated in momentum.
- **Quantitative prediction, so this can fail sharply:** the energy-phase ladder gave
  `P_C3a/P_conf ∝ Lw^−0.74` (measured `dlnR2/dlnLw = +0.200`, exactly Weaver; `dlnPb/dlnLw = +0.44`
  vs Weaver's +0.40). Extrapolating from B3M's measured `ratio@entry` = 0.1227 at `Lw` = 1 predicts
  **0.68 / 0.12 / 0.054 / 0.022** at `Lw` = 0.1 / 1 / 3 / 10. This is an *extrapolation out of the
  regime it was fitted in* — Weaver scalings do not hold at the handover — so a miss falsifies the
  extrapolation, not C3c.
- **Resolution caveat, registered up front:** `ratio@cross` lands at 1.03–1.47 rather than 1.0
  because snapshots are segment-spaced. WW's crossing is bracketed by 3 snapshots, so a lag below
  ~2% of `t_entry` is at the resolution limit and must not be reported as a resolved ordering.

**B3M LADDER RESULT (2026-08-13) — neither registered branch is right, and the truth is a third
thing.** All four rungs completed, all four reach transition *and* momentum, none void. Artifacts:
`data/b5s3_ladder_lag.csv`, `data/b5s3_ladder_regime.csv`, `data/b5s3_ladder_screen.csv`; tool
`harness/lag_vs_handover.py`.

| `Lw` | `t_entry` | `t_cross` | lag | lag/`t_entry` | %tr dur | ratio@entry | confined frac of transition |
|---|---|---|---|---|---|---|---|
| 0.1 | 0.66317 | 0.67284 | +0.00968 | +1.5% | 7.4% | 0.7144 | 8.8% |
| 1 | 0.27277 | 0.30121 | +0.02844 | +10.4% | 21.5% | 0.1227 | 23.8% |
| 3 | 0.17799 | 0.21229 | +0.03430 | +19.3% | 26.6% | 0.0553 | 28.9% |
| 10 | 0.11617 | 0.15935 | +0.04318 | +37.2% | 32.9% | 0.0235 | 38.8% |

**The quantitative prediction held, out of the regime it was fitted in.** Predicted `ratio@entry`
0.68 / 0.12 / 0.054 / 0.022; measured 0.7144 / 0.1227 / 0.0553 / 0.0235 — errors +5.1% / anchor /
+2.4% / +6.8%, and the fitted exponent comes out **−0.743** against the pre-registered −0.74. The
Weaver-derived scaling survives to the handover. **This is direct evidence *against* (b)'s
conclusion**: C3a's normalization is not too generous by an O(1) factor, it is right.

**But `t_cross` FALLS with wind** (0.673 → 0.159), which is literally (b)'s trigger, and the
momentum phase is **100% HII-dominated on all four rungs**, which is (b)'s other trigger. Both
triggers fire while (b)'s conclusion is false — so the registered dichotomy was mis-specified, not
merely unresolved.

**The mechanism, from the phase durations — this is the part neither branch anticipated:**

| `Lw` | energy dur | implicit dur | transition dur | momentum dur |
|---|---|---|---|---|
| 0.1 | 0.0030 | 0.6577 | 0.1314 | 0.7034 |
| 1 | 0.0030 | 0.2673 | 0.1322 | 1.0930 |
| 3 | 0.0030 | 0.1725 | 0.1291 | 1.1909 |
| 10 | 0.0030 | 0.1107 | 0.1311 | 1.2528 |

The energy phase lasts **exactly 0.0030 Myr at every wind strength**, and the transition phase is
**wind-independent** (0.129–0.132, exponent −0.002). Everything that moves is the **implicit** phase,
which collapses as `Lw^−0.388` (0.658 → 0.111 Myr). So `t_cross` falls with wind purely because
stronger winds reach the handover sooner — not because wind fails to confine.

**Wind domination is real, strongly wind-sensitive, and lives in the transition phase.** Measured as
the confined fraction of transition — the quantity that is *not* contaminated by the handover moving
— it grows monotonically **8.8% → 38.8%** across two decades of wind, and the lag grows +1.5% →
+37.2% of `t_entry`. That is the registered (a) evidence, and it passes.

**Momentum is the part that does NOT resolve.** `P_C3a/P_ram` in momentum falls only as `Lw^−0.33`
(max branch; −0.40 on the min branch), so reaching unity would need `Lw ≈ 260` (≈ 51 on the min
branch) — far outside any physical wind strength. The C3c momentum drive does weaken steeply with
wind (7.1× stock at `Lw` = 0.1 → 1.7× at `Lw` = 10), but never inverts. **C3c predicts a
photoionisation-dominated momentum phase universally**, and the ladder shows this is not fixable by
re-normalising C3a — the exponent, not the prefactor, is what keeps momentum HII-dominated. The
Lancaster tension is therefore *resolved in transition and open in momentum*.

⚠️ **Retraction (2026-08-13), recorded because it was reported before being checked.** Mid-run I
read the energy-phase *row counts* (69/87/96/105 here, 97/105/115 on the void ladder) as "stronger
winds hold the run in the energy phase longer". **That is wrong.** The energy-phase *duration* is
identical (0.0030 Myr) across all four rungs; the extra rows are timestep refinement, because
stronger winds are stiffer. Row count is not duration, and the two point in opposite directions here
— measured in time, stronger winds leave the early phases *sooner*. Any future reading of phase
occupancy in this workstream must use durations.

**Stage 2 RESULT (2026-08-13) — C3c runs clean on 5/5, no fate changes, and the offline screen
predicted it.** Arm `harness/b5s2_c3c.patch`; ledgers `data/b5s2_c3c_ledger.csv` (matched-`t`) and
`data/b5s2_c3c_arm_regime.csv` (the arm's own regime structure).

| config | ΔR2 max | ΔR2 end | fate | t_end | R2_end (stock → C3c) |
|---|---|---|---|---|---|
| PRB | 12.97% | **0.73%** | stopping_time → same | 0.100 → 0.100 | 0.462 → 0.459 |
| B1M | 12.93% | 5.61% | stopping_time → same | 1.5 → 1.5 | 40.45 → 42.72 |
| B2M | 12.77% | 11.61% | stopping_time → same | 1.5 → 1.5 | 29.60 → 33.04 |
| WW | 17.06% | 17.06% | **shell_collapsed → shell_collapsed** | 0.2816 → **0.2358** | 0.897 → 0.997 |
| B3M | 20.52% | 20.52% | stopping_time → same | 1.5 → 1.5 | 19.29 → 23.25 |

- **Zero numerical distress** on every config — no excess-work, overflow, monotonic-guard or
  convergence warnings. **The `t_cross` kink did not trouble the integrator**, so the §3c.1 remedy
  (a solver event via `phase_events.py`) is *not* needed on this evidence. Keep it registered in case
  a stage-3 config behaves differently.
- **No fate changed.** WW still collapses, 16% earlier (0.2816 → 0.2358 Myr) — the stronger
  photoionised drive does not save it, it reorders the collapse. Under D3 this is a *timing* change,
  not a fate change.
- **Every config is OVER-BAR (12.8–20.5%), which was pre-registered as expected**, not a failure:
  the momentum drive rises 2.4–4.3× and the early-energy drive falls up to 3.3×. All five R2_end
  values move *outward* except PRB (which barely moves, 0.73%, because it never leaves implicit).

**The falsifiable null passed exactly.** Pre-registered before the runs: implicit-phase `P_HII` must
be 0 on every row and `P_drive` must equal `Pb` exactly. Measured: `P_HII > 0` on **0 of 330 implicit
rows** across all five configs, `P_drive == Pb` to machine precision on every one. In the energy
phase `P_drive` equals the **ramped** bubble pressure on 86/87, 75/76, 64/65 rows (the one outlier
per run is the 1a→1b handoff) — the D-ramp fix, working as designed.

**The offline screen predicted the self-consistent arm closely** — the methodological result worth
keeping:

| quantity | screen (on stock trajectories) | arm (self-consistent) |
|---|---|---|
| frac HII-dom, energy / implicit | 0.0000 / 0.0000 | **0.0000 / 0.0000** |
| frac HII-dom, transition B3M | 0.7619 | **0.7619** (exact) |
| frac HII-dom, transition B2M/B1M | 0.8085 | **0.8043** |
| frac HII-dom, momentum | 1.0000 | **1.0000** |
| `t_cross` B3M / B2M | 0.301207 / 0.449094 | **0.301207 / 0.449094** (exact) |
| `t_cross` B1M / WW | 0.665536 / 0.163115 | 0.686245 / **0.116782** |
| PRB crossover | never | **never** |

B3M, B2M and PRB reproduce to the printed digit; B1M drifts 3%; WW moves 28% earlier, which is
expected — WW is the only config whose *trajectory* changed enough to move its own crossover. **A
cheap offline screen on stock trajectories is a trustworthy filter for this class of change**, which
is what makes stage 3 affordable.

**Implementation fidelity confirmed independently:** re-running the C3c screen *on the arm's own
output* returns a drive ratio of 1.000 in every phase of every config — the shipped arm computes
exactly what §3c specifies.

**Verdict: C3c passes stage 2 on numerics and on structure.** It does **not** yet have a physics
verdict — that needs D3 (is a 16%-earlier collapse acceptable?) and stage 3's regime map, which is
where the Lancaster tension gets settled.

**Stage 3 (proposed 2026-08-13, from a maintainer question): the regime map — schemes where `Pb`
dominates over `P_HII`.** Under stock the ordering is frozen: the cap makes `P_HII ≤ Pb` an identity,
so "`Pb` dominates" is true by construction and carries no information. **Under C3c the ordering is
physics**, and both regimes are reachable:

- **Already in hand:** the confined branch *is* the `Pb`-dominated regime, and the current screen
  shows it holding through 100% of energy and implicit rows in all five configs — and through PRB's
  **entire run** (never crosses; compact, dense, confined throughout). See `b5_c3c_regime.csv`.
- **Strong-wind rung** (the mirror of WW): `FB_thermCoeffWind ∈ {3, 10}` on the SC cloud. Prediction:
  `t_cross` moves later or out of the run entirely, and if the momentum phase is reached while
  `P_C3a ≤ P_ram`, the drive there is `P_ram` alone — the wind-dominated coevolution branch of the
  clarified table above. Offline-screenable *after* a cheap b1-style run of each rung.
- **Low-`Qi` corner**: low-sfe clouds (F1HI's sfe = 0.01 is already committed but collapses early; a
  low-sfe *diffuse* variant would live longer in the ionized phases).
**Stage 3 partial result — B3ML (2026-08-13): the second-crossover prediction is FALSIFIED.**
B3M run to `stop_t` 5, past the bundled table's ~3.6 Myr SN onset. Out to 5 Myr there is **exactly
one** regime flip — confined → HII-dominated at t = 0.3012 Myr, in transition, matching B3M's
`t_cross` to the digit. It never flips back.

| t [Myr] | `Qi/Qi₀` | `Lmech/L₀` | R2 [pc] | `P_C3a/P_conf` |
|---|---|---|---|---|
| 0.207 | 1.01 | 1.01 | 4.4 | 0.10 (confined) |
| 0.349 | 1.02 | 1.03 | 5.9 | **4.58** (flipped) |
| 2.09 | 1.22 | 1.16 | 27.8 | 6.30 |
| **3.59** (SN onset) | 1.17 | **2.09** | 53.0 | **3.03** |
| 5.00 | **0.50** | 1.71 | 77.9 | 3.61 |

**The prediction was directionally right and quantitatively wrong.** SN onset does what §3c said:
`Lmech` doubles and the ratio **halves**, 6.30 → 3.03; `Qi` does fade, to half its initial value by
5 Myr. Both push toward confinement. **But R2 expansion beats them**: `P_C3a ∝ R2^(−3/2)` falls more
slowly than `P_ram ∝ R2^(−2)`, so the ratio *grows* with radius, and by SN onset the shell is at
53 pc. The geometric tailwind exceeds the SN wind boost; the ratio bottoms at 3.0 and turns back up.

**What this changes:** the late-time re-confinement channel is not available at this cloud's
expansion rate. It is not ruled out in general — a config where the shell is still small at SN onset
(dense, confined, slowly expanding) would have far less geometric tailwind to overcome. But it is no
longer a headline C3c prediction, and any future claim of it must name the R2 at SN onset. ⛔ as a
general prediction; open only for the small-R2-at-SN-onset corner.

- ~~**Late-time `Qi` fade**~~ (⛔ tested above, falsified for B3M-like expansion): `stop_t 15` on B3M/SC. Past SN onset (~3.6 Myr in the bundled table) the
  ionizing output collapses while winds+SNe keep `Lmech` high, so C3c predicts a possible **second
  crossover back to confinement** — `P_C3a` falling under `P_ram` late, the drive relaxing to `P_ram`
  alone. Stock cannot represent this at all; it is a C3c-only, falsifiable prediction, and the
  cheapest genuinely new physics this workstream could buy next.

**Stage 2 — 🔷 IN FLIGHT (launched 2026-08-13).** Arm = `harness/b5s2_c3c.patch`; five configs:
B2M, B3M, WW (momentum-reaching) plus PRB and B1M. Gated exactly as Batch 3 was —

⚠️ **Controls are weaker here than in Batch 3 — do not read them the same way.** For C1, B1M was a
*provably inert* control: C1 touched only the momentum phase and B1M never enters it, so ΔR2 had to
be 0.000% and was. **Under C3c nothing is inert**, because the energy phase changes too (the D-ramp
fix — the confined branch returns 0.0, so `max(Pb_eff, 0) = Pb_eff` replaces stock's un-ramped
`P_HII`). PRB and B1M are therefore *isolation* controls, not null controls: neither reaches
momentum, so they isolate C3c's energy/implicit half from its momentum half. The offline screen
predicts their energy-phase drive ratio at 0.30–0.94, i.e. **they should move, and a 0.000% result
would mean the arm is not wired in.** The falsifiable null that does survive is narrower: the
*implicit* phase, where the screen predicts exactly 1.000 — any implicit-phase deviation beyond the
~3e-14 cross-worktree noise floor is a bug in the arm, not physics.


matched-`t` ledger, fates enumerated under D3. Expect ΔR2 well over the 5% bar by construction
(momentum drive ×2.4–4.3, early-energy drive ÷ up to 3.3); the questions the arm answers are whether
fates flip, whether the integrator tolerates the transition handover kink (§3c.1: pre-registered remedy is an
event via `phase_events.py`, not a smoothing constant), and what the fate map
looks like vs stock. Implementation note for the arm: replace the `n_IF_Str` → `P_HII` path with the
§3c branch — smallest diff is computing `P_C3a` in the phase runners next to the existing `P_HII`
lines and selecting per §3c's table; the cap and shell structure stay untouched (they still serve
absorption fractions and diagnostics).

### Batch 6 — land — Status: 🟡 **C3c LANDED (`c43a50e`, PR #738); verification incomplete**

**Done:** C3c is the default at all six `P_HII` call sites, pinned by `test/test_phii_c3c.py`
(11 tests). The 13-config matched-`t` ledger is complete on both arms —
`data/b6_ledger.csv` — with **no fate change on any config**; ΔR2_max 7.6–20.5%. Two rows do not
mean what they look like: **SDHS changed phase structure** (stock hands over to
transition/momentum, C3c stays energy-driven to `t`=1.5; needs a maintainer read under D3), and
PRB's 5661% is a collapse-floor artifact (both arms hit the 0.01 pc floor; C3c *delays* collapse
by 56%). **Tooling gap this exposed — FIXED 2026-08-16.** `compare_trajectories.py` compared the terminal
*fate* but not the *phase sequence*, so this whole class of change was invisible to it. It now
emits `phases_base`/`phases_new` and a `PHASE-CHANGE` verdict (which also gates the exit status),
plus a `floor_grid_pct` column that labels the PRB-style collapse-floor artifact instead of
reporting it as a divergence. Pinned by `test/test_phii_comparator.py`.

**Still owed:** full `pytest` on merged `main`; goldens re-baselined under D4 with the before/after
table (`test_phase_boundary`, `test_betadelta_hybr_stress`, `test_scheme_screen`; note
`test_run_smoke` is **not** on D4's list and needs its own sign-off); CHANGELOG entry; fold-back
notes to `feature/threeway-pt2` and `feature/low-winds-regime`.

**Scope caveat:** both b6 arms ran on `bugfix/phii-pt1` code (`fca7d88` stock, `2199699` C3c), so
the ledger is a clean stock-vs-C3c comparison at a fixed base but is **not** a statement about
`main`. Batch 7 re-ran the B3M pair on current `main` and reproduced the b6 row exactly
(`R2_end` 19.2942 → 23.2527), so main's energy-collapse event, betadelta tolerances and exact sps
spline are neutral **for that config** — the other twelve are unretested against `main`.

Original scope for this batch: chosen candidate (D1 decides between C1, C1⊕C2b, or a C3) on the
**full-12**; full ladder
re-verify; CHANGELOG entry; reconcile the evidence README (§7 answers), DOC_STATUS, and — when
the sibling branches merge — fold-back notes for momentum-pdrive (its §2 "inferred" caveat, its
CSV column rename) and weak-winds (quantitative collapse times now clean). Goldens re-baselined
under D4 with a table of before/after.

### Batch 7 — confinement coverage + the weak-wind flip — Status: ⬜

**Registered 2026-08-14 BEFORE any run of this batch started. Nothing below was written or
edited after results existed.**

Motivation. Every claim that the energy and implicit phases are 100% confined rests on **5
configs with a real C3c arm** (B3M, WW, B2M, PRB, B1M — `data/b5s2_c3c_arm_regime.csv`) plus 4
*offline* wind rungs evaluated on the stock trajectory. **Eight of the thirteen** matrix configs
(SC, F1HI, F1LO, GMC, BE, PL2, LDLS, SDHS) have never been regime-screened at all. The confined
fraction is a property of a C3c run **alone** — no stock arm is needed — so the gap costs 8 runs,
not 16.

The margin is not marginal: on B3M the ratio `P_C3a/P_conf` peaks at **0.120** in energy and
**0.107** in implicit (`data/b7_regime_trajectory.csv`), i.e. **8.3×** and **9.4×** below the
switch. The governing scaling, from Weaver (`R2 ∝ (Lw/ρ)^{1/5} t^{3/5}`,
`Pb ∝ Lw^{2/5} ρ^{3/5} t^{−4/5}`) with `P_C3a ∝ Qi^{1/2} R2^{−3/2}`:

```
P_C3a / Pb  ∝  Qi^0.5 · Lw^−0.7 · ρ^−0.3 · t^−0.1
```

`Qi` and `Lw` both scale ≈linearly with cluster mass, so the ratio goes as **M^−0.2** — nearly
mass-independent, which is why it barely moves across the mass/sfe grid; and `ρ^−0.3` puts the
*denser* configs further from the switch, not closer.

**G7.1 — coverage (the null).** On all 8 never-screened configs, the C3c arm returns
`frac_HII_dom = 0.0000` in **both** energy and implicit.
- FALSIFIED IF any of the 8 shows a non-zero energy or implicit HII-dominated fraction.
- Registered secondary: `ratio_max` in energy stays **< 0.5** on all 8 (B3M is 0.120; the
  spread across 4 decades of `nCore` should not cost 4×).

**G7.2 — the flip (the control that proves the null CAN be non-null).** A new rung `B3MW001`
(`FB_thermCoeffWind = 0.01`, `Qi` untouched — this decouples wind from ionising output, which no
mass/sfe change can do) **DOES** break confinement in the energy phase.
- From `Lw^−0.7`: the ratio scales by `0.01^−0.7 = 25.1×`. B3M energy max **0.120 → 3.01**,
  median **0.072 → 1.81**.
- Registered: energy `frac_HII_dom` **> 0.5**, and energy `ratio_max` in **[1.5, 6.0]**.
- FALSIFIED IF energy `frac_HII_dom = 0`. That outcome would mean the Weaver scaling does not
  govern the energy phase — which would **also remove the basis for G7.1's inference** to the
  unmeasured configs, so the two gates stand or fall together.
- Anchor already in hand: the same law predicts `B3MW01` (`Lw × 0.1`) at `0.120 × 5.01 = 0.60`,
  i.e. still confined — and the committed offline screen reports exactly `frac_HII_dom = 0.0000`
  for `B3MW01` energy. The law is therefore already right at one rung out.

**VOID rule (the stage-3 lesson, §3c).** If a run terminates before leaving the implicit phase,
or fails to complete, its row is reported **VOID** — never as a confirming null. A null is only
evidence here because G7.2 is expected to produce a non-null on the same screen.

Artifacts: `data/b7_confinement_screen.csv`, `harness/screen_confinement.py`.

### Batch 8 — the photo-only limit: Spitzer / Hosokawa–Inutsuka cross-check — Status: ⬜

**Registered 2026-08-16 BEFORE the harness was written or run. Nothing below was written or
edited after results existed.**

Motivation. §3's candidate table put a **two-sided** obligation on C3: it "must reproduce limiting
cases (wind-only → Weaver-like, photo-only → Spitzer-like)". The **wind-only half was discharged**
by Batch 5 stage 3 — `ratio@entry` followed `Lw^−0.743` against Weaver's predicted −0.74, errors
2–7%, out of the fitted regime. The **photo-only half has never been checked**, and it is the half
that bears on the one question this workstream still has open: C3c predicts a photoionisation-
dominated momentum phase in *every* configuration measured (`P_C3a/P_ram` = 3.5–7.6; inversion
needs `Lw ≈ 260`). The shipped docstring asserts this is "NOT an O(1) normalisation error". That
assertion currently rests on a *consistency* argument — the same normalisation predicts the
transition crossover to 7% — not on an external anchor. Spitzer is the external anchor.

This batch needs **no solver run**: it is closed-form in the shipped helper.

**The target.** Classical D-type expansion into a uniform medium, thin swept-up shell, no wind:

```
d/dt (M R') = 4 pi R^2 P_HII ,   M = (4/3) pi R^3 rho_0 ,   P_HII = rho_0 c_i^2 (R_St/R)^{3/2}
```

`P_C3a ∝ R^{−3/2}` is exactly the Strömgren scaling, so the equation is self-similar with
`R = A t^{4/7}`. Matching amplitudes gives `A = [(49/12) c_i^2 R_St^{3/2}]^{2/7}`, which is
*identically* the large-`t` limit of Hosokawa & Inutsuka (2006),
`R = R_St [1 + (7/4) sqrt(4/3) c_i t / R_St]^{4/7}`. Spitzer (1978)'s ram-balance closure
(`rho_0 R'^2 = P_HII`) gives the same 4/7 index with amplitude lower by `(4/3)^{2/7} = 1.0855`.
So the momentum-equation integration must land on **HI**, 8.55% above Spitzer — the two classical
results bracket the answer and the gate can tell them apart.

**G8.1 — the Strömgren anchor (algebra).** The shipped `get_phii_c3c` cavity density at
`R2 = R_St ≡ (3 Qi / (4 pi chi_e alpha_B n_0^2))^{1/3}` equals the ambient `n_0`.
- Bar: relative error **< 1e-12**. FALSIFIED IF above.

**G8.2 — the pressure normalisation (algebra).** `P_C3a(R_St) = (2 + x_He(1+Z_He_shell)) n_0 k_B T`
= `rho_0 c_i^2` with `c_i^2 = (2 + x_He(1+Z_He_shell)) k_B T / (mu_convert m_H)`. For pure hydrogen
this is Spitzer's `2 n k T`; the shipped `mu_convert/mu_ion_shell` must *be* that particle count.
- Bar: relative error **< 1e-12**. FALSIFIED IF above.

**G8.3 — the expansion index (dynamics).** Integrating the thin-shell momentum equation with the
shipped helper on its driving branch, `dlnR/dlnt → 4/7 = 0.571429` by `R/R_St = 10`.
- Bar: **within 1%**. FALSIFIED IF outside.

**G8.4 — the amplitude (dynamics; the real gate).** The integrated `R(t)` matches the HI closed
form within **5%** over `R/R_St ∈ [2, 10]`, and sits **above** Spitzer by `(4/3)^{2/7}` within 5%.
- FALSIFIED IF the integrated amplitude misses HI by >5%. That outcome would mean C3a's magnitude
  does **not** reduce to the classical D-type result, and the shipped docstring's "not an O(1)
  normalisation error" claim would be **retracted** — the momentum-phase dominance would then be
  a normalisation artifact after all, and C3a's prefactor would need re-deriving.

**G8.5 — the mutation control (what makes G8.4 evidence).** G8.1–G8.4 are *expected* to pass; a
passing null is only evidence if the same gate demonstrably fails on a wrong normalisation. A
deliberately mis-normalised variant that drops the `mu_convert/mu_ion_shell` factor (2.2× low in
pressure, so `(1/2.2)^{2/7} = 0.7935`, i.e. 20.6% low in radius) **must FAIL G8.4**.
- FALSIFIED IF the mis-normalised variant passes — the gate would then be insensitive to exactly
  the class of error it exists to catch.

**Scope, stated up front.** The cross-check idealises the *test*, not the code: uniform ambient
density, no wind, no gravity, no dust absorption (`f_abs = 1`), fully swept-up thin shell. That is
the setting in which Spitzer and HI are derived, and it is the only setting in which "reproduces
the classical result" is a decidable claim. It therefore validates C3a's **magnitude and scaling**;
it says nothing about the density profile or the wind coupling, which the ladder already covered.

Artifacts: `data/b8_spitzer_crosscheck.csv`, `harness/spitzer_crosscheck.py`,
`test/test_phii_c3c_spitzer.py`.

**RESULT (2026-08-16) — C3a reproduces Hosokawa–Inutsuka exactly. G8.4 failed as written;
the defect was in the gate, not the code.**

| gate | measured | bar | verdict |
|---|---|---|---|
| G8.1 Strömgren anchor | `2.220e-16` | < 1e-12 | ✅ PASS |
| G8.2 normalisation | `3.331e-16` | < 1e-12 | ✅ PASS |
| G8.3 index → 4/7 | `0.0324%` (slope 0.57124 vs 0.571429) | < 1% | ✅ PASS |
| **G8.4 as registered** | **9.511%** | < 5% | ❌ **FAIL** |
| G8.4′ amended (below) | **0.0000%** | < 5% | ✅ PASS |
| G8.5 mutation control | `20.140%` vs analytic `20.170%` | > 5% | ✅ PASS |

**Why G8.4 failed, and why that is not a physics result.** I registered "the integrated `R(t)`
matches the HI closed form within 5% over `R/R_St ∈ [2,10]`" and integrated *from rest*. But HI's
closed form does not start from rest — differentiating it at `t=0` gives `v = sqrt(4/3) c_i`. So
the comparison measured the **startup transient**, not the amplitude. Measured decay of that
transient, from rest:

| `R/R_St` | 2 | 5 | 10 | 20 | 50 | 100 | 150 |
|---|---|---|---|---|---|---|---|
| dev vs HI | −9.51% | −2.29% | −0.70% | −0.21% | −0.04% | −0.01% | −0.01% |
| local index | 0.4985 | 0.5606 | 0.5683 | 0.5705 | 0.5712 | 0.5714 | 0.5714 |

It decays monotonically to zero and the index converges on 4/7 — the signature of an initial
condition relaxing onto an attractor, not of a wrong pressure. **G8.4 is recorded as failed as
written and amended, not reinterpreted.**

**G8.4′ (amended).** Compare like with like: integrate from HI's *own* `t=0` state. Then any
residual is a pressure error and nothing else. Measured deviation is **0.0000% at every radius
sampled over `R/R_St ∈ [2,50]`**, on all five `(n_0, Qi)` combinations — the shipped
`get_phii_c3c`, driven through the thin-shell momentum equation, does not merely approximate
Hosokawa–Inutsuka, it **is** its solution. The run simultaneously sits **8.56%** above Spitzer,
against the analytic `(4/3)^{2/7} = 8.55%`, so it lands on the momentum-equation closure and is
cleanly separated from the ram-balance one.

**The amendment does not weaken the gate** — that was checked, not assumed. Under G8.4′ the
mis-normalised control still misses by **−20.14%** against its analytic `−20.17%`, and
`test_phii_c3c_spitzer.py` was mutation-checked against `P_C3a × 1.05` and `× 1.001`: both fail.
The check resolves a **0.1%** pressure error.

**What this establishes, and what it does not — stated because the headline invites overreading.**
HI's law is *derived from* the same momentum equation with `P = rho_i c_i^2 (R_St/R)^{3/2}`. So once
G8.1 and G8.2 hold, the ODE integration **must** return HI; it is not independent confirmation.
The substantive content is therefore in the **algebra**, not the dynamics:

- G8.1 says the density C3a inverts is the Strömgren density — the same balance, not a lookalike.
- G8.2 is the real check, and it is the one that could have gone wrong: the prefactor could have
  been 1 (ions only), 2 (pure hydrogen), or a `mu` confusion in either direction. It is the
  He-correct **2.2** = `2 + x_He(1 + Z_He_shell)`, so `P_C3a` is `n_tot k T` exactly.
- The dynamical gates are then a **propagation** check — that nothing further enters between the
  helper and the shell's equation of motion — plus a demonstration, via G8.5 and the `× 1.001`
  mutation, that the chain is sensitive to the error class it is asked about.

That is weaker than "an independent test of C3a" and stronger than "algebra restated": it closes
the specific worry that C3a's *magnitude* was mis-set, which is what the open momentum question
needed ruled out.

**Consequence for the open momentum question.** `get_phii_c3c`'s docstring asserts the
photoionisation-dominated momentum phase is "NOT an O(1) normalisation error", and until now that
rested on the internal consistency argument (the same normalisation predicts the transition
crossover to 7%). **That assertion now has an external anchor and is CONFIRMED**: C3a's magnitude
is exactly the classical D-type pressure, to the precision at which a 0.1% error is detectable. So
the universal HII-dominated momentum phase is *not* a prefactor bug, and re-deriving C3a's
normalisation is a dead end. What remains open is what Batch 5 stage 3 already isolated — the
`R2^{−3/2}` cavity geometry against `P_ram ∝ R2^{−2}` — which is a **model-structure** question
(does a real momentum-phase cavity stay Strömgren-filled?), not a calibration one.

Both halves of §3's limiting-case obligation on C3 are now discharged: wind-only → Weaver-like
(Batch 5 stage 3, exponent −0.743 vs −0.74) and photo-only → Spitzer-like (here, exact).

### Batch 9 — the geometry question — Status: 🟡 **G9.2 FALSIFIED, G9.3 discharged (2026-08-17)**

> ⛔ **THE SCOPING HEADLINE BELOW WAS WRONG AND IS RETRACTED.** The scope (2026-08-16) concluded
> "the geometry correction is strictly one-signed — it makes `P_HII` larger, never smaller — so
> geometry cannot be the escape hatch for the HII-dominated momentum phase." That was measured on
> energy/implicit/transition rows only, with momentum flagged as uncovered (G9.3). **The B3M
> momentum run (2026-08-17) falsifies it in the momentum phase**: `ratio > 1` on **0 of 34**
> momentum rows (0.505–0.705), because the shell there is **thick** (`dR/R2` = 0.670–1.308, i.e.
> past the `dR = R2/3` break-even where the layer volume exceeds the cavity volume). Geometry
> *lowers* `P_HII` in momentum by 0.51–0.71×. The surviving part of the conclusion is weaker and
> stated in the verdict below. Read the verdict, not the scope.

**Scope as originally written (2026-08-16), kept for the record.** The measurement was a screen over
*already-committed* Batch 7 run output (no new runs), written to motivate and bound the work.

**The question.** C3a takes the density that balances recombination over the **whole cavity**,
`n_C3a = sqrt(3 Qi_abs / (4 pi chi_e alpha_B R2^3))`. But trinity's own shell solve puts the
photoionised gas in a **thin layer at the inner edge of the shell** (`shell_structure.py` integrates
`nShell_arr_ion(r)` up to `shell_ion_idx`); the cavity interior holds hot/wind gas, not photoionised
gas. And `Qi_abs = Qi * shell_fAbsorbedIon` is the photon budget **absorbed in the shell**. So C3a
takes the photon budget of the layer and spreads it over the volume of the cavity. Those are
different volumes holding different gas.

For a layer of thickness `dR` at `R2`, balancing over `4 pi R2^2 dR` instead of `(4/3) pi R2^3`:

```
n_layer / n_cavity  =  sqrt( R2 / (3 dR) )        > 1  for any  dR < R2/3
```

and `P` is linear in `n`, so that is also the pressure ratio.

**Measured (`data/b9_geometry_scope.csv`, `harness/geometry_screen.py`, 1085 rows, 9 configs):**

| | energy | implicit | transition |
|---|---|---|---|
| `dR/R2` min | 5.7e-06 … 5.5e-04 | 6.2e-05 … 6.9e-02 | 3.1e-03 |
| ratio median | 6.7 … 100.7 | 1.75 … 40.9 | 10.4 |
| ratio max | 24.7 … 240.8 | 2.21 … 73.1 | 10.4 |

**The correction is strictly one-signed: `ratio > 1` on 1.0000 of all rows**, in every phase of
every config, median 1.75–100.7. **So making C3a's geometry agree with trinity's own shell solution
makes `P_HII` LARGER, by one to two orders of magnitude — never smaller.**

**This reframes the open momentum question.** "Does the momentum-phase cavity really stay
Strömgren-filled?" is *not* the escape hatch for the universally HII-dominated momentum phase — the
alternative geometry deepens the dominance. C3a is best read as a **conservative lower bound** on the
photoionised pressure, which means the resolution must come from the **pressure coupling**, exactly
as `c43a50e`'s own commit message flagged: *"a photoevaporative flow does not drive at n k T of the
whole region, which is unexplored."* That is now the load-bearing unknown, and it is a **D-question**
(D5 below), not a measurement.

⚠️ **Correction made while scoping, recorded because it was reasoned from before being checked.**
From a single last-row snapshot per config I read `shell_fAbsorbedIon = 1.000` everywhere and
inferred "no ionising photon is left to maintain a cavity H II region, in any config". **Across full
trajectories that is false**: `frac_fabs_ge_099` runs from **0.000** (B3MW001/F1LO/LDLS energy)
to 1.000, so escaping fractions are substantial in the low-density and weak-wind configs. It does
not change the volume-mismatch result above (escaping photons ionise nothing, and `Qi_abs` already
carries `f_abs`), but the "all photons absorbed in the shell" premise is withdrawn.

**Pre-registered gates, none discharged.**

- **G9.1 — photon bookkeeping.** Per phase per config, report `frac_fabs_ge_099` and the escaping
  fraction. No pass/fail: this is the input inventory G9.4 needs, and the scoping run shows it is
  strongly config-dependent, so it must be measured rather than assumed.
- **G9.2 — the correction is one-signed.** `n_layer/n_cavity > 1` on every row. **Screened at
  1.0000 on 1085 rows (energy/implicit/transition)**; registered so momentum can falsify it.
  FALSIFIED IF any row in any phase returns ≤ 1, which would mean a geometry correction that
  *lowers* `P_HII` exists and the reframing above is wrong.
- **G9.3 — momentum coverage (the gap that matters).** ⚠️ **The scoping run does not cover the
  momentum phase at all** — Batch 7's configs never reached it, and B3M is the only config in the
  matrix that does. No conclusion about momentum geometry may be drawn until G9.2 is measured on
  momentum rows. This is the one thing that needs a run, and it needs exactly one.
- **G9.4 — the layer model, computed not scaled.** The ratios above come from the analytic volume
  scaling. Before any of this informs a code change, recompute the layer density through the same
  shell machinery (`nShell_arr_ion`, `shell_ion_idx`) rather than from `sqrt(R2/3dR)`, and confirm
  the two agree to within a stated tolerance. FALSIFIED IF they disagree by more than 2×, which
  would mean the thin-layer scaling is not the right idealisation of trinity's own solve.

**Explicitly out of scope.** This does not propose changing C3a's geometry. The measurement's
purpose is the opposite — to show that the geometry lever moves `P_HII` the wrong way, so effort
should go to D5 instead of to a volume refactor.

---

**VERDICT (2026-08-17) — G9.2 FALSIFIED in momentum; G9.3 discharged. The mechanism is shell
thickness, and it breaks the scope's one-signedness claim exactly where the open question lives.**

One B3M run at `stop_t` 1.5 (`--arm b9`, code `2fa8cc9c`, clean tree), 231 rows, all four phases,
`R2_end` = 23.253 — reproducing the Batch 0 / Batch 5 stage 2 B3M trajectory (231 snapshots,
`R2_end` 23.25), so provenance is anchored.

| phase | rows | `dR/R2` | ratio `n_layer/n_cavity` | frac ratio > 1 |
|---|---|---|---|---|
| energy | 87 | 0.000 … 0.002 | 11.78 … 128.87 | **1.0000** |
| implicit | 68 | 0.002 … 0.006 | 7.72 … 11.83 | **1.0000** |
| transition | 42 | 0.007 … **0.668** | 0.707 … 6.853 | **0.3810** |
| momentum | 34 | **0.670 … 1.308** | **0.505 … 0.705** | **0.0000** |

**Mechanism.** The ratio is `sqrt(R2/(3 dR))`, so it crosses 1 at `dR = R2/3`. The shell is thin
early (`dR/R2` ~ 1e-3) and **thick** in momentum (`dR/R2` ≥ 0.67, exceeding 1 by late times) — past
break-even the ionised-layer volume `4 pi R2^2 dR` *exceeds* the cavity volume `(4/3) pi R2^3`, so
the layer density is **lower** than C3a's cavity density. The crossover happens **inside the
transition phase** (`dR/R2` runs 0.007 → 0.668 within those 42 rows), which is why transition comes
out mixed at 38.1%. So the correction is one-signed *per phase*, in opposite directions, not globally.

**What survives, and what does not.**

- ⛔ **Retracted:** "geometry deepens the HII dominance, so it cannot be the escape hatch." False in
  momentum. Geometry *softens* it there, by 0.51–0.71×.
- ✅ **Survives, and is the useful result:** the correction **does not flip the regime**. Measured on
  the same rows, `P_HII/P_ram` has median **6.165** (range 5.083–7.161); layer-corrected it is median
  **3.594** (range 3.584–3.614), and **34 of 34 momentum rows remain HII-dominated**. So a geometry
  fix alone cannot produce the wind-dominated momentum branch Lancaster's work implies — it removes
  about 40% of the excess and leaves a factor ~3.6.
- 🔍 **Unregistered observation worth keeping:** the layer-corrected ratio is nearly **constant in
  time** (3.584 → 3.614 over t = 0.405 → 1.5) while the uncorrected one *climbs* 5.08 → 7.16. The
  cavity form's growing dominance is an artifact of holding `dR` fixed in the geometry while `R2`
  grows; the layer form predicts a time-independent dominance factor. Flagged, not concluded — one
  config, and `dR/R2` crossing 1 needs its own physical scrutiny (a shell thicker than its own
  radius is at the edge of the thin-shell idealisation both C3a and the ODEs assume).

**Consequence for D5.** D5 remains the live question, but for a *weaker* reason than the scope
claimed: not "geometry pushes the wrong way" but "geometry moves the right way and is not enough."
The `Lw ~ 260` inversion requirement from Batch 5 stage 3 becomes roughly `Lw ~ 260 x 0.6^(1/0.33)`
if the layer form is adopted — still far outside physical wind strengths. Pressure coupling is still
where the resolution has to come from.

**G9.1 (inventory, no pass/fail).** `frac_fabs_ge_099` on B3M: energy 0.241, implicit 1.000,
transition 1.000, momentum 0.765 — so escape is significant in the energy phase and in ~24% of
momentum rows. Strongly phase- and config-dependent, as the scope's correction already noted.

**G9.4 — CLOSED 2026-08-17: FALSIFIED at 3.17× against a 2× bar.**
`harness/layer_density_check.py` replays the shipped `shell_structure_pure` on each snapshot's own
state, so `shell_ion_idx` and `shell_r_arr` share an index space and the arrays are the original
un-downsampled ones. Artifact `data/b9_layer_density.csv` (B3M, stride 2, 116 rows replayed, 0
failures).

| phase | n | `dR_ion/R2` | ion frac of shell | ratio (analytic) | ratio (profile) | **rms/analytic** | recomb/`Qi_abs` |
|---|---|---|---|---|---|---|---|
| energy | 44 | 0.0004 | 1.0000 | 31.99 | 15.43 | **0.496** | 0.247 |
| implicit | 34 | 0.0025 | 0.8741 | 11.50 | 10.27 | **0.906** | 0.823 |
| transition | 21 | 0.5015 | 0.9653 | 0.815 | 0.402 | **0.493** | 0.385 |
| momentum | 17 | 0.9751 | 0.9954 | 0.585 | 0.251 | **0.429** | 0.393 |

Range 0.315–0.939, worst disagreement **3.171× > 2×** ⇒ **FALSIFIED**. The analytic thin-layer
Strömgren scaling is *not* a valid stand-in for the real profile; it **overestimates** the
recombination-equivalent density everywhere, by up to 3.2×.

**Mechanism, and it is exact where the layer is thin.** A Strömgren balance assumes every absorbed
ionising photon is consumed by *recombination*; the real shell also absorbs them on **dust**. The
measured recombination integral takes only `recomb/Qi_abs` = 0.247 / 0.823 / 0.385 / 0.393 of the
budget, and density scales as its square root:

| | `sqrt(recomb/Qi_abs)` | measured `rms/analytic` |
|---|---|---|
| energy | 0.497 | **0.496** |
| implicit | 0.907 | **0.906** |

— agreement to three decimals, so in the thin-layer phases the entire G9.4 gap **is** the dust
sink. In transition/momentum (0.620 vs 0.493, 0.627 vs 0.429) a second factor appears: the layer is
thick, so the `r^2` weighting across a real density gradient no longer reduces to a single uniform
density. Both are reasons the idealisation fails, not defects in the code.

⚠️ **[B11.0 2026-08-18 — the layer volume itself is a thin-shell approximation, and momentum is not
thin.]** `layer_density_check.py:140` uses `V_lay = 4πR2²·dR_ion` where the exact spherical shell is
`(4/3)π((R2+dR)³ − R2³)`; the ratio is `1 + x + x²/3` with `x = dR_ion/R2 ≈ 0.98` in momentum, so
**V_exact/V_thin = 1.802–2.878 (median 2.292)** and every quantity built on `n_layer_analytic` is
overstated by **√that = 1.342–1.696**. On the exact volume the momentum `ratio_analytic` becomes
**0.2976–0.5302** (from 0.5049–0.7118) — G9.2's verdict is unchanged and *strengthened*, but the
numbers below are the thin-form ones. Re-fit registered as B11.F. Checked, not assumed: Batch 10's
`Lw^−0.1133` exponent and `Lw ≈ 46.5` inversion were fitted to `pdrive_profile`, which does **not**
use `V_lay`, so they are unaffected — the bias reaches only `n_layer_analytic`, `ratio_analytic`,
`pdrive_analytic` and `rms_over_analytic`.

**G9.2's momentum falsification SURVIVES the recheck**, which was the live worry. With the *true*
ionised thickness: momentum ratio **0.505–0.712, frac > 1 = 0.0000** — statistically identical to the
0.505–0.705 reported from the clamped value. The reason is now measured rather than assumed: the
shell is **99.54%** ionised in momentum, so `dR_full ≈ dR_ion` there (and 87.4% in implicit, a 13%
`dR` error worth ~7% in the ratio). ⚠️ **The Batch 9 screen's `dR` was nevertheless the full shell
thickness, not the ionised layer** — `shell_ion_idx` (up to 26848) indexes the original arrays while
the snapshot's `shell_r_arr` is downsampled to ≤100 points, so the index clamped on 100% of rows.
Benign here, but `geometry_screen.py` now says so in its docstring and reports the metric as
`frac_index_clamped` rather than the misleading `frac_whole_shell_ionised`.

**⚠️ This supersedes Batch 9's headline number, and moves it a long way.** Batch 9 reported the
corrected momentum dominance as `P_HII/P_ram` ≈ **3.594** using the analytic layer form. Using the
**profile** form — the one G9.4 shows is the trustworthy one — the same 17 momentum rows give
**median 1.545, range 1.322–1.666**, and it **falls monotonically with time** (1.666 at t=0.407 →
1.322 at t=1.500). Still HII-dominated on 17/17 rows, so no fate or regime claim changes; but the
excess over unity is ~50%, not ~260%.

**🔍 Lead worth testing, flagged as an extrapolation and NOT a result.** Batch 5 stage 3 measured
`P_C3a/P_ram ∝ Lw^−0.33` and concluded inversion needs `Lw ≈ 260`. Applying that exponent to the
⛔ **TESTED AND DEAD — Batch 10 falsified this lead (2026-08-17).** The profile form does *not*
inherit the cavity form's `Lw^−0.33`: measured on `B3MW3`/`B3MW10` it falls only as `Lw^−0.1133`,
because stronger winds thin the shell (`dR_ion/R2 ∝ Lw^−0.3375`) and that *raises* the geometry
correction. Revised inversion `Lw` ≈ **46.5**, still unphysical. The paragraph below is kept as the
registered reasoning that got tested; read §Batch 10 for the outcome.
profile-corrected 1.545 gives inversion at **`Lw ≈ 3.4`** — an entirely physical wind strength, and
`B3MW3` / `B3MW10` already exist in the matrix (`run_batch.py` `b3mladder`). If that survives a real
test it would **resolve the Lancaster tension** rather than deepen it. Caveats that keep this a lead:
one config; the −0.33 exponent was fitted to the *uncorrected* form and may not carry; and the
profile form is a diagnostic, not a shipped prescription. **The honest next measurement is the
profile-based ratio on the existing `B3MW3`/`B3MW10` rungs, not a code change.**

**Reproduce.** `python docs/dev/phii-identity/harness/run_batch.py --arm b9 --configs B3M
--stop-t 1.5 --root <scratch>/runs/b9`, then
`python docs/dev/phii-identity/harness/geometry_screen.py <scratch>/runs/b9/B3M/ <scratch>/runs/b7/*/
--out docs/dev/phii-identity/data/b9_geometry_scope.csv`.

⚠️ **Wall time is not in `data/b9_walltimes.csv`.** `run_batch.py` crashed on
`root.relative_to(REPO)` *after* the run finished, because `--root` pointed outside the repo — the
runs survived, the CSV did not, and the re-invocation that wrote it recorded `status=skipped` with
blank timing. The bug is fixed in this commit. Wall time derived from launch/last-snapshot mtimes:
**~590 s (9.8 min)**, consistent with Batch 0's 682 s for the same config.

Artifacts: `data/b9_geometry_scope.csv` (all 4 phases, 10 configs), `harness/geometry_screen.py`,
`data/b9_layer_density.csv` + `harness/layer_density_check.py` (G9.4),
`data/b9_walltimes.csv` (timing lost, see above).

**G9.4 reproduce:** `python docs/dev/phii-identity/harness/layer_density_check.py
<scratch>/runs/b9/B3M/ --stride 2 --out docs/dev/phii-identity/data/b9_layer_density.csv`
(~2 s — it replays the shell solve, it does not integrate the ODEs).

### Batch 10 — does the profile form invert on the strong-wind rungs? — Status: ⬜

**Registered 2026-08-17 BEFORE either run of this batch started. Nothing below was written or
edited after results existed.** Point predictions are stated to 4 dp so they cannot be
retro-fitted.

Motivation. Batch 9 + G9.4 measured the momentum-phase drive ratio on B3M (`Lw` = 1) under the
**profile form** — the ionised-layer volume with the real dust-depleted recombination — at
**median 1.545** (1.322–1.666), against the shipped cavity form's 6.165. Stage 3 independently
measured `P_C3a/P_ram ∝ Lw^−0.33` on the `B3MW01/1/3/10` ladder. Composing the two puts the
wind/photoionisation crossover at **`Lw` = 3.74**, i.e. *inside* the ladder that already exists,
where stage 3's own conclusion was that inversion needs `Lw ≈ 260`. If that holds, the
wind-dominated momentum branch Lancaster's coevolution work implies is **reachable**, and D5 stops
being the only route out.

This is the measurement that promotes the Batch 9 lead to a result or kills it.

**What is being tested, stated precisely.** The profile-based ratio is a **diagnostic**, not shipped
physics. A crossover here would mean *a layer+dust-corrected model* inverts; it would **not** mean
trinity's momentum phase inverts as shipped. The shipped cavity form is expected to stay
HII-dominated on both rungs (G10.3 is the control for exactly that).

| gate | prediction | bar |
|---|---|---|
| **G10.1** `B3MW3` profile median | `1.545 × 3^−0.33` = **1.0752** | in **[0.85, 1.35]** |
| **G10.2** `B3MW10` profile median | `1.545 × 10^−0.33` = **0.7227** | **< 1.0**, and in **[0.55, 0.95]** |
| **G10.3** control: shipped cavity form | HII-dominated on 100% of momentum rows, both rungs | matches stage 3 |
| **G10.4** monotonicity | profile median falls with `Lw`: B3M > B3MW3 > B3MW10 | strict |

- **G10.2 is the falsifiable one. FALSIFIED IF `B3MW10`'s profile median ≥ 1.0** — that would mean
  stage 3's `Lw^−0.33` does not carry to the profile form, the `Lw` = 3.74 crossover estimate is
  **void**, and Batch 9's lead is withdrawn. I expect this to be the likeliest failure mode, because
  the exponent was fitted to the *cavity* form and the layer form's `dR(Lw)` dependence is unmeasured
  — stronger winds may thin the shell, which would push the ratio back up.
- **VOID rule (§3c stage-3 lesson).** If a run terminates before entering the momentum phase, its
  row is **VOID**, never a confirming null. Both rungs reached momentum in stage 3, so a failure to
  do so is itself reportable.
- **G10.5 — `dR/R2` inventory.** Report the momentum-phase `dR_ion/R2` per rung. B3M sits at 0.975,
  right at the thin-shell idealisation's limit; if the strong-wind rungs sit far from it the
  comparison across rungs is not like-for-like and must be said so.

Artifacts: `data/b10_wind_profile.csv`, reusing `harness/layer_density_check.py` unchanged.

---

**VERDICT (2026-08-17) — G10.2 FALSIFIED. The `Lw` = 3.74 crossover is void, and the registered
failure mechanism is confirmed quantitatively.** Both rungs ran clean to `stop_t` 1.5 and both
reached momentum (`B3MW3` 239 rows / 514 s, `B3MW10` 248 rows / 400 s). Their energy row counts,
**96 and 105**, reproduce stage 3's recorded 96/105 for these rungs exactly, so provenance is
anchored to the earlier ladder.

| run | `Lw` | `dR_ion/R2` | ratio (analytic) | frac ratio>1 | ratio (profile) | `pdrive_cavity` | **`pdrive_profile`** |
|---|---|---|---|---|---|---|---|
| B3M | 1 | 0.9751 | 0.5847 | 0.0000 | 0.2506 | 6.1646 | **1.5451** |
| B3MW3 | 3 | 0.7370 | 0.6806 | 0.0000 | 0.3313 | 4.0485 | **1.3412** |
| B3MW10 | 10 | 0.4483 | 0.8785 | **0.1667** | 0.4512 | 2.4773 | **1.1902** |

| gate | predicted | measured | verdict |
|---|---|---|---|
| G10.1 `B3MW3` in [0.85, 1.35] | 1.0752 | **1.3412** | ✅ PASS — *by 0.0088*, and 25% above the point prediction |
| G10.2 `B3MW10` < 1.0, in [0.55, 0.95] | 0.7227 | **1.1902** | ❌ **FALSIFIED** |
| G10.3 control: cavity form HII-dominated | 100% | **1.0000 both rungs** | ✅ PASS — matches stage 3 |
| G10.4 monotonic in `Lw` | strict | 1.5451 > 1.3412 > 1.1902 | ✅ PASS |
| G10.5 `dR_ion/R2` like-for-like? | — | 0.9751 / 0.7370 / 0.4483 | ⚠️ **NOT like-for-like** |

**The pre-registered failure mode is exactly what happened.** G10.2 registered: *"the exponent was
fitted to the cavity form and the layer form's `dR(Lw)` dependence is unmeasured — stronger winds may
thin the shell, which would push the ratio back up."* Measured power laws in `Lw` (1 → 10):

```
pdrive_cavity  ∝ Lw^-0.3959      (stage 3 measured -0.33 on this ladder)
dR_ion/R2      ∝ Lw^-0.3375      <- stronger winds DO thin the shell, ~ Lw^-1/3
sqrt(R2/3 dR)  ∝ Lw^+0.169       <- so the geometry correction RISES with wind
pdrive_profile ∝ Lw^-0.1133      <- net: the correction cancels ~43% of the cavity decline
```

So the layer+dust correction **flattens the wind dependence** rather than inheriting it. My Batch 9
extrapolation assumed the profile form would carry the cavity form's −0.33; it carries −0.113.

**Revised crossover, and it kills the lead.** Inversion (`ratio` = 1) on this ladder:

| form | inversion at |
|---|---|
| cavity (shipped) | `Lw` ≈ **99** |
| profile (corrected) | `Lw` ≈ **46.5** |
| ~~Batch 9 extrapolation~~ | ~~3.74~~ **void** |

The correction helps by ~2×, not the ~26× I projected. **`Lw` ≈ 46 is still unphysical**, so the
wind-dominated momentum branch is **not** reachable through the geometry/dust correction, and **D5
(pressure coupling) remains the live route** — the conclusion Batch 9's lead had put in doubt.
⚠️ Note my cavity exponent (−0.3959) is steeper than stage 3's (−0.33), so my cavity inversion
estimate (99) sits below stage 3's (260); both are extrapolations one to two decades outside the
fitted range, which is itself reason to treat either number as an order-of-magnitude statement only.

**🔍 A real refinement of Batch 9's G9.2 verdict — the geometry correction's sign is
wind-dependent.** `frac(ratio > 1)` in momentum is 0.0000 / 0.0000 / **0.1667**: `B3MW10`'s
`dR_ion/R2` reaches down to **0.3197**, dipping below the `dR = R2/3` break-even, so 3 of its 18
momentum rows have the correction *raising* `P_HII` again. **This reconciles the Batch 9 scope with
the Batch 9 verdict.** The scope said the correction is one-signed and raises `P_HII`; the verdict
said it lowers it in momentum. Both were partial views of a **thickness-dependent sign**, and shell
thickness tracks wind strength — so neither statement generalises, and the honest form is: the sign
is set by `dR_ion` vs `R2/3`, which is a function of `Lw`.

**G10.5 caveat, registered in advance and now binding.** The three rungs sit at substantially
different shell thicknesses (0.98 → 0.74 → 0.45), so the cross-rung profile trend is **confounded
with the thickness trend** and the −0.1133 exponent is not a clean wind response. It is the right
number for "what does the corrected model predict across this ladder", and the wrong number for
"how does photoionised pressure respond to wind at fixed geometry". The `Lw` ≈ 46.5 figure inherits
that confound.

**Also holds across rungs:** the near-time-independence of the analytic layer form noted in Batch 9
generalises — `pdrive_analytic` spans 3.5945–3.6237 (B3M), 2.6946–2.7719 (B3MW3), 2.0385–2.1450
(B3MW10), i.e. flat to ~1–3% within each rung while the cavity form varies 40% within B3M alone.
⚠️ **[B11.0 2026-08-18 — this claim does NOT survive the exact layer volume.]** `n_layer_analytic`
uses the thin-shell `4πR2²·dR` (see B11.0 S1); on the exact spherical-shell volume the same rows give
**2.1298–2.6993** (B3M, median 2.3807), **1.8315–2.1730** (B3MW3, 1.9942), **1.5917–1.7524**
(B3MW10, 1.7165) — B3M spans **±12%**, not 1–3%, so the near-time-independence claimed in this
paragraph is itself an artefact of the thin-shell volume, and is **withdrawn**. **Everything else in
Batch 10 survives**: G10.1–G10.4, the `Lw^−0.1133` fit and the `Lw ≈ 46.5` inversion are all gated
on / fitted to `pdrive_profile`, which is built from `n_rms_profile` (the real shell profile) over
the exact cavity volume and never uses `V_lay` — verified by re-deriving the exponent (0.1133) and
the inversion (46.5) from the published profile medians. Only `n_layer_analytic`, `ratio_analytic`,
`pdrive_analytic` and `rms_over_analytic` carry the bias. The one narrative casualty is the
*mechanism* sentence above: the `sqrt(R2/3dR) ∝ Lw^+0.169` cancellation is stated in the thin-shell
correction, so its exponent is thin-form; the measured `dR_ion/R2 ∝ Lw^−0.3375` underneath it is
raw data and stands.

**Reproduce:** `python docs/dev/phii-identity/harness/run_batch.py --arm b10 --configs B3MW3,B3MW10
--stop-t 1.5 --root <scratch>/runs/b10`, then `layer_density_check.py <scratch>/runs/b9/B3M/
<scratch>/runs/b10/B3MW3/ <scratch>/runs/b10/B3MW10/ --stride 2 --out
docs/dev/phii-identity/data/b10_wind_profile.csv`.

Artifacts: `data/b10_wind_profile.csv`, `data/b10_walltimes.csv`.

### Batch 11 — verify, then quantify, the four driving-branch seams — Status: 🟡 **B11.0 DONE (2026-08-18): 3 CONFIRMED, 1 REVISED, 0 REFUTED; A–D not started**

**Maintainer ruling (2026-08-18):** the seams in the audit below "look like real problems that
prevent code shipping, and we need to investigate further. Do not assume that they are correctly
evaluated." Both halves are binding: the seams block further P_HII shipping, **and the audit's own
claims are unverified** — they were produced in one session, from one config's runs, with no
adversarial pass. B11.0 exists to falsify them before B11.A–D spend anything on them.

**B11.0 — adversarial re-verification of the audit itself (MANDATORY FIRST; do not skip to A–D).**
Re-derive each seam from current source and data as if trying to kill it. Known weak points of the
original analysis, per seam:
- **A (photon double-spend).** Claim rests on `phi0 = 1` at `shell_structure.py:120` and on no
  upstream depletion of `Qi`. Re-check: grep every consumer of `Qi` between SPS output and the shell
  solve — is there ANY cavity-absorption factor applied anywhere (e.g. in `update_feedback`,
  `read_sps`, or the phase runners) that the audit missed? Also confirm the hot cavity really is
  treated as transparent (no code path attenuates ionising flux across R1→R2).
- **B (boundary-pressure mismatch).** Claim rests on `nShell0 ∝ params['Pb']` at
  `shell_structure.py:125`. Re-check which `Pb` that is *at call time* in each phase runner (order
  of assignment vs the shell call — momentum assigns `params['Pb'] = P_ram` at `:585,669,891`), and
  whether any runner passes a drive-consistent pressure instead.
- **C (mass double-book) — highest priority and highest risk of being wrong, because it is a
  units-sensitive derived number.** The 0.095→0.564 table inverts the shipped `P_HII` back to
  `n_C3a` via `(mu_convert/mu_ion_shell)·k_B·T` and converts `(4/3)πR2³·n·mu_convert` to Msun in
  internal units. Units are this repo's declared recurring bug class: **run the `units-reviewer`
  agent over the derivation**, cross-check `n` against the committed `n_cavity` column in
  `data/b10_wind_profile.csv`, and sanity-check against an independent route (e.g. cavity mass from
  `n_cgs · V_cgs · mu_convert · m_H` in cgs). Also verify `shell_mass` on those rows really is the
  full swept cloud mass (B3M at R2 = 23 pc vs its `rCloud`; `shell_mass` barely grows — confirm why).
- **D (thin-shell strain).** Twice-derived already (Batch 9 clamped + G9.4 replay agree); lowest
  risk. Re-confirm `dR_full` vs `dR_ion` on a spot-check row.
Verdict per seam: **CONFIRMED / REVISED (with the corrected number) / REFUTED**, each with the
command or line reference that decides it. A REFUTED seam is dropped from A–D and the audit section
is corrected in place, dated.

#### B11.0 RESULT — 2026-08-18. Verdicts: A CONFIRMED · B **REVISED** · C CONFIRMED (strengthened) · D CONFIRMED

**Nothing was taken on the audit's word.** Every number below was re-derived from current source, a
fresh B3M run at `ef624195` (`--arm b9 --configs B3M --stop-t 1.5`, 495.9 s, 231 snapshots, all four
phases, `R2_end` = 23.253 — reproduces Batch 5/9 exactly), or the committed CSVs. The fresh run
reproduces `data/b9_layer_density.csv` to **≤3.3e-06 relative on all 116 rows × 15 numeric columns**,
so the Batch 9/10 baselines are sound and the seams are not a run artefact.

| seam | verdict | what decides it |
|---|---|---|
| **A** photon double-spend | **CONFIRMED** | no cavity-absorption factor exists anywhere between SPS and either consumer; total claimed budget = `2·Qi·f_abs` |
| **B** boundary-pressure mismatch | **REVISED** | the mismatch is real (`P_HII/Pb` median **6.16** momentum, **4.62** transition) but the audit's *direction* is wrong: the loop back into `P_C3a` is **exactly zero on 88% of driving rows** and **upward**, not downward, on the rest |
| **C** mass double-book | **CONFIRMED, and stronger than claimed** | 0.095 → **0.5638** reproduced to 4 s.f. by two independent routes agreeing to 1e-12; and the shell already holds **100.0000%** of the gas the run has, so `M_cav` has no supply at all |
| **D** thin-shell strain | **CONFIRMED** | `dR_full/R2` = **0.6723–1.3078**, `dR_ion/R2` = **0.6579–1.3076** on B3M momentum |

**A — photon double-spend: CONFIRMED.** The PLAN asked for a grep of every `Qi` consumer between
SPS output and the shell solve. Complete list in `trinity/`: `sps/read_sps.py:254` (table load),
`sps/update_feedback.py:164` (interpolate), `shell_structure/shell_structure.py:108,145,244`,
`shell_structure/get_shellODE.py:90`, `bubble_structure/get_bubbleParams.py:352`,
`bubble_structure/bubble_luminosity.py:428,779,812`, `phase1_energy/energy_phase_ODEs.py:146`.
**No cavity-absorption factor at any of them.** `params['Qi']` is written verbatim by
`updateDict` (`_input/dictionary.py:1264-1269`) at every driving-branch call site
(`run_momentum_phase.py:578,890`; `run_transition_phase.py:497,837`) — the dataclass carries the raw
SPS value and nothing rescales it. The cavity is transparent: the only attenuation in the code is
`get_shellODE.py:120` (`dphidr`), integrated outward from `rShell_start = R2` with `phi0 = 1`
(`shell_structure.py:119-120`), so nothing attenuates across R1→R2.
`bubble_luminosity.py:428,779,812` do use an undepleted `phi = Qi/(4πr²)` *inside* the cavity, but
only as the radiation-field axis of the cooling/heating tables — no photons are removed, so it is
not a fourth spend (and it is energy/implicit-phase code, i.e. confined branch).
Sharper than the audit put it: `f_abs = shell_fAbsorbedIon = 1 − f_esc_ion`
(`shell_structure.py:401,457`) is by construction the fraction absorbed **by the shell**, and
`get_bubbleParams.py:358` then spends that identical sub-budget on **cavity** recombinations. It is
not "the same photons twice" in a loose sense — the cavity is credited with exactly the photons the
shell already ate. Measured `f_abs` on driving rows: **1.0000 on 16/16 transition and 13/17
momentum** rows, so the claimed budget is `≈2·Qi` — twice what the star emits.
*External corroboration (Geen et al., "When H II Regions are Complicated", §4):* "the UV photons
from the star **are not absorbed by the wind bubble**" — so trinity's transparent cavity is the
standard picture and is **correct**; what is not standard is the balance volume (see C).

**B — boundary-pressure mismatch: REVISED.** The mismatch itself is confirmed exactly as described.
`nShell0 ∝ params['Pb']` at `shell_structure.py:125-126`. Call-time ordering checked in every
driving-branch runner: momentum assigns `params['Pb'] = pRam(...)` at `run_momentum_phase.py:585`,
**before** `shell_structure_pure(params)` at `:626` and `get_phii_c3c` at `:636`, and re-assigns at
`:669` and `:891`; transition assigns `params['Pb'] = Pb` (thermal, from `compute_R1_Pb`) at
`run_transition_phase.py:509` before the shell call at `:558`/`get_phii_c3c` at `:566`, and at
`:842` before `:843`/`:848`. **No runner passes a drive-consistent pressure.** Magnitude measured on
the fresh run: `P_HII/Pb` median **6.1646** (5.091–7.156) in momentum, **4.6218** (1.302–5.067) in
transition — the audit's "≈6×" is right where the momentum question lives.
**What is wrong is the direction.** §6b's summary lists B under "every seam pushes the same way …
the shipped `P_C3a` is an upper bound". Measured, `f_abs` = 1.0000 on **100% of transition driving
rows and 76.5% of momentum driving rows** — already saturated, so raising the inner boundary
pressure cannot change `Qi_abs`, and the loop back into `P_C3a` is **exactly zero on 29 of 33
driving rows (88%)**. On the 4 rows where `f_abs` < 1 (0.7208–1.0000, t ≥ 1.29), a higher `nShell0`
means *more* absorption ⇒ higher `Qi_abs` ⇒ **higher** `P_C3a`. So B is not an upper-bound
mechanism; its live consequences are thickness, dust column and gravity sampling.
*And B11.B's premise needs re-aiming.* Geen et al. §4.2 close the same system with
`P_w = n_i c_i² m_H/X` **at `r_w`** — i.e. the wind pressure *is* the correct inner boundary
pressure for the photoionised gas, which is exactly `nShell0 ∝ P_ram`. The shipped boundary
condition is therefore the standard one, and `P_C3a` is a *second, contradictory value for the same
quantity*. B11.B as registered ("set the inner pressure to the drive's claim") would replace the
defensible side with the questionable one. Re-scoped below.

**C — mass double-book: CONFIRMED, and stronger than the audit stated.** Highest-risk item, so it
was attacked four ways.
1. *Units.* The `units-reviewer` agent was run over the derivation as the PLAN required. Verdict:
   the inversion is the exact algebraic inverse, `n` comes back as a **hydrogen-nuclei** density in
   pc⁻³, the 2.2 (`mu_convert/mu_ion_shell`) is on the correct side (dividing by it is required —
   dividing by `k_B T` alone would return `n_tot` and overstate the mass by exactly 2.2), and
   `mu_convert` is the right multiplier for `M = V·n·μ` (`mu_ion_shell` would be wrong by 1/2.2).
   Decisive corroboration: `shell_structure.py:125-126` **is** the shipped inverse of the same map,
   so the audit used the code's own convention rather than reinventing one.
2. *Two independent routes.* `harness/mass_ledger_check.py` computes `n` both by inverting the
   shipped `P_HII` (the audit's route) and by replaying the forward `get_phii_c3c` map through
   `shell_structure_pure` (independent of `P_HII` entirely).
   **route P / route Q = 1.000000000000 on all 33 driving rows.** A units or algebra error in the
   inversion would have shown up here; none did. This also disposes of a live worry that the
   momentum phase-boundary reconciliation snapshot (`run_momentum_phase.py:888-896`) pairs a stale
   `P_HII` with a fresh `R2`/`Pb`: it never recomputes `P_HII`, but empirically no driving row in
   this run carries a stale one.
3. *Cross-check against the committed `n_cavity` column.* `data/b10_wind_profile.csv`'s `n_cavity`
   is built from `Qi_abs` by `layer_density_check.py`, never from `P_HII`. Feeding it through
   `M_cav = (4/3)πR2³·n·mu_convert` gives **57,396.6 Msun at t=1.5** — the audit's 57,400, from a
   third route, before the new run existed.
4. *Physical sanity.* n_cavity at t=1.5 is 31.5 cm⁻³ at R2 = 23.25 pc ⇒ `Qi ≈ 4.9e50` s⁻¹, the right
   order for this 1e4 Msun cluster.

**Reproduced numbers** (`data/b11_mass_ledger.csv`): `M_cav/M_shell` = **0.0952** at t=0.4074 (first
momentum row) → **0.5638** at t=1.5, i.e. **57,397 vs 101,805 Msun**. Audit claimed 0.095 → 0.564,
57,400 vs 101,800. Agreement to 4 significant figures on every quoted figure.

The PLAN also asked to confirm `shell_mass` really is the full swept cloud mass, and why it barely
grows. **Confirmed, with the mechanism:** `rCloud` = 4.999 pc and the cloud's gas is 100,000 Msun
(`read_param` splits the input `mCloud` 110,000 into `mCloud` 100,000 + `mCluster` 10,000 — note it
is already post-SF, so `(1−sfe)` must *not* be applied again). All of it is swept by t = 0.3037.
Beyond `rCloud` only `nISM` = 1 cm⁻³ remains, contributing **1,805 Msun** out to R2 = 23.25 pc — so
`shell_mass` goes 100,005 → 101,805 (**+1.8%**) while R2 grows **4.65×**.

**Three findings the audit did not have, all pushing the same way:**
- **The shell already holds 100% of the gas that exists.** `shell_mass / (mCloud + swept ambient)` =
  **0.999997–1.000000** on every driving row. There is no unbooked reservoir for `M_cav` to come
  from — it is not that the books disagree, it is that one book is already full.
- **The model asserts more gas than the simulation has.**
  `(M_cav + shell_mass) / M_avail` = **1.5638** at t=1.5 (and already 1.0717 at t=0.3037). That is a
  strictly stronger statement than a double-book: total ionised gas would be 159,202 Msun drawn from
  a 101,805 Msun supply.
- **Winds cannot supply it either.** Independent budget from the run's own feedback columns,
  `∫ 2·Lmech_total/v_mech_total² dt` over 0→1.5 Myr = **54.8 Msun**, i.e. **1/1047 of `M_cav`**.
  (Do **not** use the snapshot's `bubble_mass` for this: the momentum phase never recomputes it, so
  it is frozen at 99.643 Msun on all 34 momentum rows. Confirmed by inspection.)
- **Onset is in transition, not momentum.** Over-subscription begins at the first driving row
  (t = 0.3037, ratio 1.0717), so this is not a momentum-phase-only concern.

*External corroboration:* Geen et al. §4.2 write photoionisation equilibrium with a wind bubble as
`(4π/3)·n_i²·(r_i³ − r_w³)·α_B = Q_H` — the balance volume **excludes** the wind bubble, and the
photoionised gas lives between `r_w` and `r_i`. C3a balances over `(4/3)πR2³`, which in trinity's
geometry is `(4/3)π r_w³` — *exactly the volume the standard treatment excludes*. So the cavity mass
C3a implies is not merely unfunded in trinity's ledger; on the published picture it should not exist.

**D — thin-shell strain: CONFIRMED.** Spot-checked as the PLAN asked, then done over all rows.
B3M momentum: `dR_full/R2` = **0.6723–1.3078**, `dR_ion/R2` = **0.6579–1.3076** (audit: 0.67–1.31).
`dR_ion/dR_full` median **0.9954** — the momentum shell is essentially entirely ionised, so the two
thicknesses are the same number and the seam does not depend on which is meant. Command:
`python docs/dev/phii-identity/harness/layer_density_check.py <run>/B3M/ --stride 2`.

**Side-findings from B11.0 (outside the four seams — recorded because they change committed numbers).**
- **S1 — `layer_density_check.py:140` uses a thin-shell layer volume in a regime that is not thin.**
  `V_lay = 4πR2²·dR_ion` versus the exact `(4/3)π((R2+dR)³ − R2³)`; the ratio is `1 + x + x²/3` with
  `x = dR_ion/R2`, and momentum runs at `x ≈ 0.98`. Measured **V_exact/V_thin = 1.802–2.878 (median
  2.292)**, so `n_layer_analytic`, `ratio_analytic` and `pdrive_analytic` are overstated by
  **√that = 1.342–1.696 (median 1.514)**. Corrected momentum values:
  `ratio_analytic` (Batch 9 G9.2) 0.5049–0.7118 → **0.2976–0.5302** (verdict unchanged, strengthened);
  `pdrive_analytic` (Batch 10) B3M 3.5945–3.6237 → **2.1298–2.6993** (median 2.3807),
  B3MW3 2.6946–2.7719 → **1.8315–2.1730** (1.9942), B3MW10 2.0385–2.1450 → **1.5917–1.7524** (1.7165).
  Batch 10's "flat to ~1–3% within each rung" does **not** survive — B3M spans ±12% on the exact
  form — and is withdrawn. Re-fit of the affected columns registered as B11.F below.
  **Scope of the bias, checked rather than assumed:** only `n_layer_analytic`, `ratio_analytic`,
  `pdrive_analytic` and `rms_over_analytic` use `V_lay`. `n_cavity`, `n_rms_profile`,
  `ratio_from_profile` and `pdrive_profile` do not, so **Batch 10's G10.1–G10.4 are untouched**, and
  so is every seam-C number (which runs through `n_cavity`). **G9.4's verdict also stands**: its
  worst disagreement (3.171 vs a 2× bar) sits in the *energy* phase where `dR/R2` ≈ 4e-4 and the
  correction is 1.0004. It does change the momentum picture favourably — `rms_over_analytic` there
  goes 0.368–0.460 → **0.593–0.679**, i.e. the analytic layer form and the real profile agree far
  better once the volume is exact.
- **S2 — `layer_density_check.py:154` drops zeros, not just missing values.**
  `pd_cav = (pH/pb) if (pH and pb and pb > 0)` treats `P_HII == 0.0` as absent. Rows silently
  dropped: energy 44/44, implicit 34/34, transition 5/21, **momentum 0/17**. Every `pdrive_*` figure
  this PLAN quotes is momentum-only, so **no published number is affected** — but any future
  all-phase use of that column would be biased upward. Not changed here: the harness's outputs are
  committed baselines, so a fix belongs with a regeneration, in B11.E.
- **S3 — `run_momentum_phase.py:888-896` never recomputes `P_HII`** on the phase-boundary
  reconciliation snapshot, though it does recompute `Pb`, `R1` and `shell_props`. No effect on any
  B11.0 number (see C.2). Folded into B11.E's list; any change must be shown bit-identical.
- **S4 — `data/b9_walltimes.csv` now carries a real B3M timing.** Batch 9's `--root` bug left
  `B3M,skipped,,`; the B11.0 reproduction fills it with **495.9 s / 231 snapshots**. Batch 9's
  mtime-derived "~590 s" estimate was ~19% high.

**What B11.0 changes for A–D.** A, C and D stand and are quantified below as registered. B is
re-scoped: since the shipped `nShell0 ∝ Pb` is the physically standard closure, B11.B must measure
the *inconsistency*, not "fix" the shell — see the amended B11.B. And the single most useful
outcome of B11.0 is that all four seams are **absent by construction** from Geen et al.'s two-equation
closure (their Eqs. for photoionisation equilibrium with a wind bubble, plus wind/photoionised
pressure balance at `r_w`), which solves for one `n_i` and one `r_i`. That is a concrete external
reference model for D5 rather than a from-scratch design, and is registered as **B11.G**.

**B11.A — photon-conserving cavity+shell accounting (offline, no solver run).** Iterate to a fixed
point on committed/replayed trajectories: cavity consumes what its Strömgren balance claims, shell
receives the remainder, `f_abs` recomputed via `shell_structure_pure`, `Qi_abs` updated, repeat.
Note the possible outcome that the fixed point is degenerate or bistable (cavity consumes
everything → neutral shell) — that outcome is itself the answer, not a failure. Gate: report
`P_C3a_fixedpoint / P_C3a_shipped` per phase; pre-register that energy/implicit are unchanged
(confined branch, `P_HII` = 0 either way).
**B11.B — the boundary/drive inconsistency, measured (offline replay).** ⚠️ **Re-scoped by B11.0.**
As registered this read "set the inner pressure to the drive's claim instead of `params['Pb']`" —
but B11.0 found `nShell0 ∝ Pb` is the *standard* closure (Geen et al.'s wind/photoionised pressure
balance at `r_w`), so that would have replaced the defensible side. What to measure instead: re-run
`shell_structure_pure` on driving rows at both pressures and report the *spread* — Δ`f_abs`,
Δthickness, Δdust column, Δgravity sampling — as the size of the inconsistency, not as a correction.
Pre-register: Δ`f_abs` = 0 on the 29/33 rows where `f_abs` is already 1.0000, and Δ`P_C3a` ≥ 0 (never
negative) on the remaining 4 — B11.0 predicts the direction, so a negative Δ falsifies B11.0's
revision. The replay harness pattern exists (`harness/layer_density_check.py`,
`harness/mass_ledger_check.py`).
**B11.C — mass ledger consequence (only if C survives B11.0).** Two sub-questions: (i) *supply* —
can photoevaporation off the shell actually deliver `dM_cav/dt` (compare `n_C3a·c_i·4πR2²` against
the required filling rate)? If not, the cavity is supply-limited and the driving branch overstates
`P_C3a` for a *fifth* independent reason; (ii) *dynamics* — re-integrate the momentum-phase
equation of motion offline with `shell_mass` debited by `M_cav(t)`, ΔR2 at matched `t` vs stock.
Measure, don't guess whether 56% mass matters.
**B11.D — thin-shell validity bound.** Document as a stated validity limit of the ODE + C3a split
at `dR/R2 ≳ 1`; no fix proposed this batch.
**B11.E — cleanup (trivial tier, after A–D):** the vestigial `n_IF_Str > 0` gates and the stale
"from n_IF_Str" comments in the four phase runners; plus B11.0's **S3** (`P_HII` not recomputed on
the momentum reconciliation snapshot, `run_momentum_phase.py:888-896`) and **S2** (the falsy-zero
filter in `layer_density_check.py:154`, which must be fixed together with regenerating the CSVs it
produced). Any gate change must be shown bit-identical or is deferred.

**B11.F — re-fit Batch 9/10's geometry numbers on the exact layer volume (offline, no run).**
Opened by B11.0's **S1**: `n_layer_analytic` used the thin-shell `4πR2²·dR` where `dR/R2 ≈ 1`, so it
is overstated by 1.34–1.70×. The *verdicts* of G9.2 and G10.2 survive (G9.2 moves further from 1 in
the direction already reported; G10.2 is gated on `pdrive_profile` and is untouched), and so do the
`Lw^−0.1133` fit and `Lw ≈ 46.5` inversion — both were fitted to `pdrive_profile`, checked by
re-deriving them from the published medians. What is withdrawn is the "flat to 1–3% within a rung"
claim and every `n_layer_analytic`-derived column. Scope of the re-fit: `n_layer_analytic`,
`ratio_analytic`, `pdrive_analytic`, `rms_over_analytic`, and the thin-form `sqrt(R2/3dR) ∝ Lw^+0.169`
mechanism sentence in Batch 10. Cheap: the committed `data/b10_wind_profile.csv` already carries
`R2` and `dR_ion`, so no run is needed.

**B11.G — score the shipped closure against Geen et al.'s (offline, no run).** B11.0's most useful
by-product: all four seams are absent by construction from the two-equation closure in Geen et al.,
"When H II Regions are Complicated" (§4.2) — photoionisation equilibrium over `(r_i³ − r_w³)` with a
wind bubble, closed by wind/photoionised pressure balance `P_w = n_i c_i² m_H/X` at `r_w`. One `n_i`,
one `r_i`, no double-spend (the ionised gas consumes `Q_H` once), no cavity mass (the wind bubble
holds no photoionised gas), no boundary mismatch (the pressure balance *is* the boundary condition).
Solve it on trinity's own trajectory (`r_w := R2`, `Q_H`, `ṗ_w` from the run) and compare `n_i` and
`r_i − r_w` against the shipped `n_C3a`, `nShell0` and `dR_ion`. This is a **reference-model
comparison, not a proposal to adopt it**: Geen's algebra assumes `w = 2` in places and B3M is
uniform (`densPL_alpha = 0`), so only the profile-independent equations transfer. Feeds D5 directly.

**Hold released only when:** ~~B11.0 verdicts are in for all four seams~~ (**DONE 2026-08-18**: A/C/D
CONFIRMED, B REVISED, none REFUTED) AND A–C are quantified. Then D5's key design (`phii_scheme`,
default, ramp-window handling) resumes with those numbers in hand. B11.F and B11.G are not on the
critical path for the hold, but B11.G materially de-risks D5.

### §6b Self-consistency audit of the C3c/C3a picture — 2026-08-18 (maintainer question) — ✅ **RE-VERIFIED by B11.0 (2026-08-18): A/C/D CONFIRMED, B REVISED, none REFUTED**

> **Status of this section after B11.0.** Every claim below was re-derived adversarially against
> current source, a fresh B3M run and the committed CSVs; see §Batch 11 → B11.0 RESULT for the
> deciding commands and line references. **Corrections made in place below are marked
> `[B11.0 2026-08-18]`.** Two things changed: seam B's *direction* was wrong (its feedback into
> `P_C3a` is zero on 88% of driving rows and upward on the rest, so it is not an upper-bound
> mechanism), which also breaks the "every seam pushes the same way" summary; and seam C is
> **understated** — the shell already holds 100% of the gas the run has, so the cavity mass has no
> supply at all. Seams A, C and D reproduce to 4 significant figures on every quoted number.

**Question asked:** is the current implementation idea self-consistent throughout the code, and as
physics? **Answer: on the confined branch, exactly; on the driving branch, no — four specific seams,
all pushing the same direction.** Checked against source and committed/derived run data, not asserted.

**Verified consistent (with the receipts):**
1. **Normalisation/units** — the 2.2 particles-per-H-nucleus prefactor is He-correct and exact to
   1e-12 (G8.2); the closed form is bit-reproducible from snapshots (2.2e-16, Batches 7/9).
2. **Limits** — wind-only → Weaver (−0.743 vs −0.74, stage 3); photo-only → Spitzer/HI exact (Batch 8).
3. **Decoupling** — the value contains no `Pb`, so the original relabelling bug class cannot recur.
4. **Energy bookkeeping** — the energy ODE charges `press_bubble·dV` only (`energy_phase_ODEs.py:274`),
   so photoionised work is NOT drawn from `Eb`. Correct: its energy source is the radiation field,
   continuously resupplied — no reservoir is double-billed.
5. **The confined branch is exactly self-consistent**, which is the deep argument for the 0.0:
   the cavity is hot wind gas (transparent), so the shell legitimately receives the full `Qi`
   (`shell_structure.py:120`, `phi0 = 1`); the skin sits in pressure equilibrium at `Pb`
   (`nShell0 ∝ Pb`, `:125`); the drive is `Pb` (transmitted); and no cavity gas mass is claimed.
   Every book balances.

**Driving-branch seams (transition 76% of rows, momentum 100% — and, for weak winds, energy):**
- **A. Photon double-spend.** The shell solve starts `phi0 = 1` — the full `Qi` arrives at the shell
  inner edge — while C3a simultaneously spends `Qi·f_abs` on cavity recombinations. The same photons
  are consumed twice, once per model; and `f_abs` (measured ≈1 in momentum) is itself computed from
  the shell that saw the un-depleted flux, then fed back into `Qi_abs`. A photon-conserving
  cavity+shell accounting has less than `Qi` available to the cavity ⇒ `P_C3a` overstated (√ of the
  budget), on top of G9.4's dust finding.
  **[B11.0 2026-08-18 — CONFIRMED, and sharper than stated.]** Every `Qi` consumer in `trinity/` was
  enumerated: **no cavity-absorption factor exists at any of them**, and the only attenuation in the
  code (`get_shellODE.py:120`) starts at `r = R2` with `phi0 = 1`, so nothing attenuates across
  R1→R2. `f_abs` = 1.0000 on 16/16 transition and 13/17 momentum driving rows, so the claimed budget
  is ≈`2·Qi` — twice what the star emits. Sharper: `f_abs` is *by construction* the fraction the
  **shell** absorbed (`shell_structure.py:401,457`), and `get_bubbleParams.py:358` credits the
  **cavity** with that identical sub-budget. The transparent cavity itself is correct physics —
  Geen et al. §4: "the UV photons from the star are not absorbed by the wind bubble" — so the defect
  is the double-credit and the balance volume (seam C), not the transparency.
- **B. Boundary-pressure mismatch.** The shell structure is integrated with its inner density set by
  `params['Pb']` (= `P_ram` in momentum) while the dynamics asserts `P_C3a ≈ 6×` that at the same
  interface. Thickness, dust column, `f_abs` and the gravity sampling are all computed under a
  pressure the drive says is wrong. ~~Feedback into `P_C3a` is bounded (via `f_abs ≤ 1`) but real.~~
  **[B11.0 2026-08-18 — REVISED.]** The mismatch is confirmed and sized (`P_HII/Pb` median **6.16**
  momentum, **4.62** transition; ordering verified in every runner). But the feedback into `P_C3a`
  is **exactly zero on 29 of 33 driving rows** — `f_abs` is already 1.0000 there, so a higher inner
  pressure cannot raise `Qi_abs` — and on the other 4 rows it pushes `P_C3a` **up**, not down.
  B is a real inconsistency in thickness/dust/gravity, **not** an upper-bound mechanism on `P_C3a`.
  Further: Geen et al. §4.2 close the ionised gas with `P_w = n_i c_i² m_H/X` at `r_w`, i.e.
  `nShell0 ∝ P_ram` **is** the standard boundary condition — so the questionable value is `P_C3a`,
  not the shell's.
- **C. Mass double-book — measured, and it is NOT small.** A Strömgren-filled cavity at `n_C3a`
  holds `M_cav = (4/3)πR2³·n_C3a·mu_convert`. On B3M's momentum rows (inverting the shipped
  `P_HII`): `M_cav/M_shell` = **0.095 at t=0.405 growing monotonically to 0.564 at t=1.5**
  (57,400 vs 101,800 Msun). That gas can only have come off the shell, yet `shell_mass` keeps 100%
  of the swept material — so the inertia and `F_grav` use one book while the drive premise uses the
  other. Either the cavity is filled and the shell should be up to ~2× lighter (faster), or it is
  supply-limited and `P_C3a` is overstated; the code currently asserts both. Grows as `R2^{3/2}`,
  so it is worst exactly where the momentum question lives. (Derived from the b9 run:
  `n = P_HII/((mu_convert/mu_ion_shell)·k_B·T)`, `M_cav = (4/3)πR2³·n·mu_convert` per snapshot.)
  **[B11.0 2026-08-18 — CONFIRMED to 4 s.f., and UNDERSTATED.]** Units cleared by the
  `units-reviewer` agent; two independent routes (invert `P_HII` / replay the forward
  `get_phii_c3c` map) agree to **1e-12** on all 33 driving rows; the committed `n_cavity` column
  gives 57,396.6 Msun by a third route that never touches `P_HII`. Re-measured: **0.0952 → 0.5638,
  57,397 vs 101,805 Msun.** What the audit missed: `shell_mass` already equals **100.0000%** of the
  gas that exists (cloud 100,000 + ambient 1,805 Msun), and winds inject only **54.8 Msun** over
  0→1.5 Myr — so the "either/or" above is not a genuine fork. The filled-cavity limb needs
  `(M_cav + shell_mass)/M_avail` = **1.5638**, i.e. 56% more gas than the run has, and is
  unavailable. Onset is at the **first driving row** (t = 0.3037, transition), not in momentum.
  `data/b11_mass_ledger.csv`, `harness/mass_ledger_check.py`.
- **D. Thin-shell strain.** `dR/R2` = 0.67–1.31 in momentum (Batch 9) — the ODE's thin-shell
  premise and C3a's sharp cavity/shell split are both at their validity edge there.
  **[B11.0 2026-08-18 — CONFIRMED.]** `dR_full/R2` = 0.6723–1.3078, `dR_ion/R2` = 0.6579–1.3076;
  `dR_ion/dR_full` median 0.9954, so the seam does not depend on which thickness is meant.
- Also vestigial, cosmetic + edge-behaviour only: the C3c call is gated by the OLD capped quantity
  (`n_IF_Str > 0`) in every phase runner, and the `F_HII` docstrings still say "from n_IF_Str".

**Direction and consequence.** ~~Every seam pushes the same way:~~ **[B11.0 2026-08-18 — CORRECTED.
Three of the four push the same way, not all four.]** **The shipped driving-branch `P_C3a` is an
upper bound** — A: photon budget √-overstated (the cavity is credited with exactly the photons the
shell already absorbed, so the claimed budget is ≈`2·Qi`); C: the filled-cavity limb is unavailable,
because the shell already holds 100% of the run's gas, so `P_C3a` must be the supply-limited limb;
plus G9.4's dust, −51–75% of the budget. **B does not belong on this list**: its feedback into
`P_C3a` is exactly zero on 88% of driving rows and upward on the rest, so it inflates nothing —
it is an inconsistency in thickness, dust column and gravity sampling instead. None of this is
*extra* work beyond D5 — these seams ARE D5's content: a photon-conserving, mass-conserving,
boundary-consistent cavity+shell model is exactly "what does the photoevaporative system transmit".
**B11.0 adds an external reference for that model**: Geen et al., "When H II Regions are
Complicated" (§4.2) close exactly this system in two equations — photoionisation equilibrium over
`(r_i³ − r_w³)` plus wind/photoionised pressure balance at `r_w` — and all four seams are absent
from it by construction. Registered as B11.G; it turns D5 from a from-scratch design into a
comparison against published algebra.
**For the c3c-vs-c3a-raw key:** the confined branch is the exactly-consistent one, so the more time
a scheme spends there, the more defensible it is; C3a-raw does not add any new driving-branch seam
(the branch behaves identically), but it extends exposure into the weak-wind ramp window where the
phase's own thermodynamics (a Weaver bubble) contradicts the 1e4 K cavity picture.

## 7. Decisions needed from the maintainer

| id | question | blocks | state |
|---|---|---|---|
| D1 | ✅ **ANSWERED 2026-08-12/13** — the momentum sum **is** intended, conditional on `P_HII` being genuinely its own calculation; the transition `max` is a **deliberate** smooth handover as `Pb → 0`. See §2. Open remainder: whether a better handover formulation exists, and what C1 actually costs (Batch 3, unrun at the time of writing) | Batch 3 verdict | **answered; C1 still unmeasured** |
| D2 | ✅ **ANSWERED 2026-08-12** — `P_HII` should be a real, separate pressure, treated as one unless the architecture cannot support it (then the assumption must be explicit). Consequence: the target is **decoupling**, and §3b shows the cap is not the coupling — the ionised volume is. Open sub-question for Batch 5: which decoupled formulation (C3a/C3b/C3c) | Batch 5 | **answered; formulation open** |
| ~~D2-old~~ | ⛔ superseded by the above. **WAS THE CRUX (Batch 4a).** Removal is proven *safe* — no blow-up materialises in any regime tested, including the compact probe. So the question is no longer "can we?" but "should we?": is the uncapped Strömgren pressure physically trustworthy at these ionized volumes, given it exceeds `Pb` on 100% of rows (up to 7.79×; the 3.36 quoted earlier was PRB's `blowup_max`, not the matrix max) and shifts trajectories 15–28%? No measurement can settle this; it needs the model's intent. Also confirm §2's reading that the cap was pragmatic, not a physics claim. | Batch 4b design; Batch 5; **4a landing** | **open** |
| D3 | ✅ **ANSWERED 2026-08-13** — acceptable-if-explained. Fate *flips* remain reportable, but a **timing** change under an explained mechanism is not a re-tune trigger. Applied to the standing case: WW's collapse moving 0.2816 → 0.2358 Myr (16% earlier) under C3c is **accepted** — it still collapses, and the mechanism (a stronger photoionised drive reordering the collapse) is documented in §3c stage 2. | Batch 3/4 verdicts | **answered** |
| D5 | ⬜ **OPEN, and now the load-bearing one (raised 2026-08-16, Batch 9 scope).** **What pressure does a photoevaporative ionised layer transmit to the neutral shell?** C3c drives at the full `n_tot k T` of the ionised gas. `c43a50e`'s own commit message flags this as unexplored: *"a photoevaporative flow does not drive at n k T of the whole region."* Batch 8 removed the calibration explanation (C3a's magnitude **is** the classical D-type pressure, exactly). Batch 9 removed the geometry explanation too, but **not** for the reason its scope first claimed: the correction is *not* one-signed — in the momentum phase the shell is thick (`dR/R2` 0.67–1.31) so the layer form **lowers** `P_HII` 0.51–0.71×. It just is not enough: `P_HII/P_ram` median 6.165 → 3.594 on the analytic layer form, and **→ 1.545 (1.322–1.666, falling with time)** on the profile form that G9.4 shows is the trustworthy one — still HII-dominated on every row, but by ~50% rather than ~500%. So geometry moves the right way and lands close to unity without crossing it, and this is what is left. ✅ **Batch 10 closed the one escape route that remained**: the `Lw ≈ 3.4` inversion extrapolated from stage 3's exponent was tested on `B3MW3`/`B3MW10` and **falsified** — the profile form falls only as `Lw^−0.1133` because stronger winds thin the shell and that raises the geometry correction, so inversion sits at `Lw` ≈ 46.5, still unphysical. **D5 is therefore load-bearing on measurement, not just on absence of alternatives.** Candidate answers, none measured: (i) full `n k T` — the status quo; (ii) a momentum-flux-limited transmission (the flow carries `rho v^2`, not `n k T`); (iii) C3's never-implemented option (c), `P_ram + max(P_C3a − P_conf, 0)` — transmit the confining pressure, add only the excess. **This is a physics-intent call, not a code call**, and it cannot be settled by measurement.<br><br>**Option (iii) analysed 2026-08-17 (maintainer question).** In the momentum phase `P_conf` **is** `P_ram` (`run_momentum_phase.py` assigns `Pb` = ram pressure), so option (iii) collapses to an identity — verified to machine precision on all 34 B3M momentum rows:<br>`P_ram + max(P_C3a − P_ram, 0)` ≡ `max(P_C3a, P_ram)`<br>i.e. **in momentum, option (iii) IS C1 ("transmit, don't add") applied to C3a.** C1 was measured in Batch 3 and judged "safe, but aimed at the wrong target" — but only because `P_HII ≡ P_ram` then made `max(P_HII, P_ram) = P_ram`, which *deleted* the photoionised channel. With C3c's decoupled `P_C3a` it deletes nothing, so **Batch 3's objection to C1 no longer applies**. Outside momentum they differ: transition's `P_conf` is the bubble thermal pressure, so `P_ram + max(P_C3a − Pb, 0)` is not a plain max there.<br><br>**Measured effect (B3M momentum, 34 rows):** shipped `P_drive/P_ram` median **7.095** (6.083–8.161) → option (iii) **6.095** (5.083–7.161) = a **14.1% reduction**, exactly the `+P_ram` double-count and nothing else. ⚠️ **So option (iii) fixes additivity, not dominance** — the drive is still ~6× ram because `P_C3a` dominates alone. Of the three options, **(ii) momentum-flux-limited is the only one with factor-level leverage on magnitude** (a photoevaporative flow carries `rho v^2`, not static `n k T`).<br><br>**Tension with D1, which must be settled to adopt (iii).** D1 ruled the momentum sum *intended*, conditional on `P_HII` being genuinely its own calculation — a condition C3c now satisfies, so by D1's own logic the sum is legitimate and (iii) reverses it. The counter is physical rather than bookkeeping: if the ionised skin is what contacts the shell, the wind pushes the *skin* and the shell feels the *skin*, so adding both double-counts the wind regardless of decoupling. Genuine disagreement between the two framings; needs the maintainer.<br><br>**Maintainer direction 2026-08-17:** "transmit, don't add" is favoured, with the `P_ram + P_HII` form retained as a **selectable key** and the default *possibly* moving to transmit. **No code written — two design decisions are deliberately parked with the maintainer:**<br>**(a) the key's default.** `'sum'` first is provably byte-identical and needs no re-verification, leaving the flip as its own gated change; defaulting straight to `'transmit'` is a **default flip**, which CLAUDE.md's planning protocol classes as risky/outward-facing — pre-registered gate, full-run equivalence on the stiff regimes in separate processes at matched `t`, plus D4 golden re-baselines.<br>**(b) the scope.** Momentum-only is 2 sites (`run_momentum_phase.py:265,445`), matches Batch 3's measured C1 arm, and respects D1's ruling that the transition `max` is a deliberate `Pb → 0` handover. Momentum+transition is 5 sites (adding `run_transition_phase.py:331`, `energy_phase_ODEs.py:253,385`), becoming `max(Pb, P_HII, P_ram)` — the same double-count is arguably the same defect there, but it has no prior measurement and touches a construct D1 called deliberate. The energy/implicit sites (`energy_phase_ODEs.py:388`, `run_energy_implicit_phase.py:532`) are `max(Pb, P_HII)` with no `P_ram` and are **unaffected either way**.<br>⚠️ Whichever lands, this is the case where a `.param` key **is** warranted — unlike the capped-Strömgren prescription (see §2a discussion), `sum` and `transmit` are two defensible physics choices, directly analogous to `betadelta_solver`. <br><br>**Superseded 2026-08-18 — transmit is DROPPED; the candidate is now 'C3a-raw' (maintainer direction).** Two follow-up questions from the maintainer: (1) is the confined branch's exact-0.0 right, or should the helper always return `P_C3a` and let the existing `max`/sum structure decide dominance ("if it's smaller than Pb it should just have subdominant contribution")? (2) momentum should stay *independent Strömgren pressure + `P_ram`* — **which it already is**: the switch is inert in momentum on every measured config (`P_C3a > P_ram` on 100% of momentum rows), so momentum needs no change under either scheme. Measured effect of C3a-raw (always-return) vs shipped C3c, on committed runs — reconstruction validated to 2.2e-16 on B3M's 66 driving rows; ±5% on B3MW001's (value-lag at the rapidly-evolving early steps):<br>• **B3M (nominal wind): drive-identical in energy (0/87 rows), implicit (0/68) and momentum (0/34)**; 1/42 transition rows changes (+7.9%) — and it **removes the `t_cross` kink registered in §3c.1**: stored `P_drive` jumps +6.8% across the switch row, C3a-raw is continuous (−1.0%).<br>• **B3MW001 (weak wind): 11/51 energy rows change, ×2.12 median, ×2.62 max — every one inside the `dt_switchon` ramp window (t ≤ 6.8e-5 Myr)**, where `P_C3a` is below the un-ramped `Pb` (0.49–0.94, genuinely confined) but above the *ramped* `press_bubble`. Always-return therefore re-admits the D-ramp defect class C3c fixed as a side effect — a pressure carrying the un-ramped scale winning the `max` inside the ramp window. Batch 4a showed early-window drive changes can retain double-digit ΔR2 downstream (F1LO 14.4%), so this needs the offline screen + a full-run gate on the weak-wind configs before landing.<br>Design consequences if C3a-raw is adopted: (a) the **D1 tension dissolves** — the momentum sum stays, nothing is reversed; (b) the **scope question dissolves** — C3a-raw is one change in `get_phii_c3c` (return `P_C3a` unconditionally), zero `P_drive` edits, each phase sorts itself via its existing `max`/sum; (c) the pinned exact-0.0 contract (`test_phii_c3c.py`) and Batch 5's pre-registered null (`P_HII`=0 on implicit rows) are deliberately superseded — D4-style before/after bookkeeping required. Remaining decisions: the key's default (`c3c` switch vs `c3a` raw) and how to handle the weak-wind ramp window. | the momentum-phase verdict; any further C3 work | **open — transmit dropped 2026-08-18; candidate is C3a-raw; default + ramp-window handling parked with the maintainer** |
| D4 | ✅ **ANSWERED 2026-08-13** — re-baselining authority **granted** for `test_phase_boundary.py`, `test_betadelta_hybr_stress.py` and `test_scheme_screen.py` fixtures, conditional on G3.4: every re-baseline lands with a committed before/after table and the mechanism named. A golden that moves for an *unexplained* reason is still a stop, not a re-baseline. | Batch 6 | **answered** |

## 8. Ledger (results land here — the one source of truth)

### 8.1 Batch verdicts
| batch | status | date | verdict (one line) | artifacts |
|---|---|---|---|---|
| 0 | ✅ | 2026-08-12 | **PASS** on 6/6 core. Identity holds on 100% of implicit and transition rows and 26/27 momentum rows (WW's final collapse row is stale-`Pb`) and ≥96.97% of energy rows, relΔ ≤2.9e-16, across 4 decades of nCore. B3M independently reproduces momentum-pdrive (`P_HII` vs `P_ram` = 2.39e-16 over 34 rows). Drive anatomy: implicit exactly 1, transition ≤1.998 (median 1.82), momentum exactly 2.000, energy ≤3.31. **`frac_nIFStr_eq_n0` = 1.0000 in every phase of every config** — the cap is bound everywhere, needing no diagnostic to show it | `data/b0_identity_grid.csv`, `data/b0_trajectories.csv`, `data/b0_walltimes.csv` |
| 1 | ✅ | 2026-08-12 | **PASS.** G1(i): B3M 231 + PRB 184 + WW 178 = **593 rows** exactly equal on every pre-existing key (repr compare), matching row counts ⇒ diagnostic inert; independently corroborated by the matched-t comparator returning 0.000% on both. Cap binds **100% of rows in every phase**; blow-up p99 1.06–7.79, max **7.786** (WW momentum; B3M 3.331, PRB 3.306, B1M 3.308). **Kill bar NOT tripped ⇒ C2a survives, Batch 4a authorised.** Corrects B0: sub-100% energy rows are `Pb` staleness at the 1a→1b handoff, not cap-slack | `data/b1_bitidentity.csv`, `data/b1_capmap.csv` |
| 2 | ⬜ | — | — | — |
| 3 | ✅ | 2026-08-13 | **C1 MEASURED — safe, small, and aimed at the wrong target.** Momentum-only `max(P_HII, P_ram)` (halving `P_drive` from `2·P_ram` to `P_ram` there) on 4 configs spanning weak winds, two masses and two bench radii. **All WITHIN-BAR, no fate changes:** B1M **0.000%**, B2M 1.24%, B3M 4.00%, WW 1.29% ΔR2 at matched `t`. B1M is the pre-registered falsifiable control — it never reaches momentum, so C1 must be inert there, and it is to 0.000%. The effect is small because momentum is only 12–15% of these runs. **Verdict: C1 does not break anything, but it does not do what D2 asks** — with `P_HII ≡ P_ram` in momentum, `max(P_HII, P_ram) = P_ram`, so C1 *deletes* the photoionised channel rather than decoupling it, and D1 says the sum is intended. Superseded as a fix by C3; retained as the measured cost of the double-count | `data/b3_c1_ledger.csv` |
| 4 | 🟡 | 2026-08-12 | **4a MEASURED — survives, but is not behaviourally neutral.** 4/4 configs (PRB, B3M, F1HI, F1LO) ran to their natural end, **zero** distress lines (no excess-work, overflow, monotonic-guard or convergence warnings), wall times *within* baseline (492–764 s vs 682–832 s). **No fate changed** on any config. Identity destroyed as intended: `frac_PHII_eq_Pb` = **0.0000** in every phase of every config (was ≥0.9697), relΔ now O(1) (0.06–2.55). But **every config breaches the 5% bar**: ΔR2 max 15.3–28.4%, all located inside the `dt_switchon` window (t = 1.3e-7 … 9e-6 Myr); ΔR2 at end-of-overlap 0.95% (PRB, recovers) → 14.4% (F1LO, retained). Mechanism: uncapped `P_HII` exceeds `Pb` on **100%** of rows (max 7.79× across the matrix; 3.36× on PRB) so it wins the `max`, lifting median `P_drive/Pb` from 1.0000 to 1.83 (PRB). **Verdict: C2a is numerically viable and physically consequential — not a free win. Landing it needs D2.** 4b not started | `data/b4a_ledger.csv`, `data/b4a_identity_grid.csv` |
| 5 | 🟡 | 2026-08-13 | **Stage 1 (offline screen) done — C3b ⛔ REJECTED, C3a advances.** No solver run: both candidates are closed-form in stored quantities, evaluated on the stock trajectory across 5 configs. C3b fails the pre-registered wind-only limit *structurally* — `n = n_cloud(R2)` has **no `Qi` dependence**, so switching the ionizing source off leaves its `P_HII` unchanged; it also steps 4 decades at `rCloud`. C3a is causally decoupled (`Qi`, `R2` only), has the correct `Qi → 0` limit, and gives sensible ionised densities (19–8055 cm⁻³ in momentum) — but sits uniformly **3.5–7.6× above `P_ram`** and never crosses it, i.e. predicts a photoionisation-dominated momentum phase in all five configs. **Stage 1b: C3c designed (§3c) and screened — it supersedes bare C3a.** The confined skin has no independent density (any decoupled-thickness skin is C3a × O(1), *higher*), so C3c is a regime switch: transmit when `P_C3a ≤ P_conf`, drive at `P_C3a` when above. Screened on the same 5 runs: implicit **exactly** untouched (ratio 1..1..1), D-ramp fixed as a side effect (energy ratio down to 0.30 = the ramp honoured), `t_cross` inside transition in all 4 configs that reach it, momentum drive 2.4–4.3× stock. **Stage 2 DONE: C3c runs clean on 5/5** — zero distress, no fate changes, null passed exactly (`P_HII`=0 on 0/330 implicit rows, `P_drive`==`Pb`), all OVER-BAR at 12.8–20.5% as pre-registered. WW collapses 16% earlier but still collapses. The offline screen predicted the self-consistent regime structure to the printed digit on 3/5 configs. Physics verdict still open: needs D3 + stage 3 | `data/b5_c3_screen.csv`, `data/b5_c3c_regime.csv`, `data/b5s2_c3c_ledger.csv`, `data/b5s2_c3c_arm_regime.csv` |
| 6 | 🟡 | 2026-08-14 | **C3c LANDED (`c43a50e`, PR #738) — verification incomplete.** 13-config matched-`t` ledger complete on both arms, **no fate change on any config**, ΔR2_max 7.6–20.5%. **SDHS changed phase structure** (stock hands over at `t`=0.147/0.791; C3c stays energy-driven to 1.5) — a fate-only check does NOT catch this, and `compare_trajectories.py` cannot see it because it diffs the terminal fate, not the phase sequence. PRB's 5661% is a collapse-floor artifact (C3c *delays* collapse 56%). **Full `pytest` DISCHARGED 2026-08-16** (during Batch 8, which changed no `trinity/` source): **1085 passed, 3 failed**, 16 deselected, 605 s. The 3 red are exactly the ones predicted below and nothing else — `test_run_smoke.py`, `test_phase_boundary.py` (`cool_beta` measured **0.8783952818088819** vs golden 0.888197, matching the recorded 0.878395 to the digit) and `test_mu_audit_drift.py` (site count measured **5** vs 11, matching the recorded 11 → 5). Confirmed pre-existing by re-running the two fast ones **in isolation** from Batch 8's new test file (trinity leaks module-level global state in-process, so this was checked, not assumed). Still owed: D4 goldens with before/after table, fold-back notes; `test_run_smoke` is **not** on D4's list and needs its own sign-off. CHANGELOG landed in `3590c91d`. Both arms ran at a pre-`main` base; Batch 7 re-ran B3M on `main` and reproduced the row exactly, so main's other physics is neutral **for that config only** | `data/b6_ledger.csv`, `data/b6c3c_walltimes.csv`, `data/b6stock_walltimes.csv` |
| 7 | 🟡 | 2026-08-16 | **G7.2 PASSES — the control fires, so the null is evidence.** `B3MW001` (`Lw × 0.01`, `Qi` untouched) breaks confinement in the **energy** phase: 78.4% HII-dominated, `ratio_max` **4.927** against a pre-registered `[1.5, 6.0]` and a point prediction of 3.01 from `Qi^0.5 Lw^−0.7 ρ^−0.3 t^−0.1`. **G7.1 holds on all 8 nominal-wind configs** — 100% confined in the energy phase across **five decades** of core density (1e2–1e6 cm⁻³), worst margin GMC `ratio_max` 0.173, i.e. 2.9× below the registered 0.5 bar. Energy phase *closed* on 6 of 9 (PL2/SDHS/BE still inside it — partial coverage, not a closed null); implicit/transition/momentum coverage still accumulating. Recomputation validated against the delivered branch on 231/231 B3M rows, `mismatch_rows`=0. Verdict: **`P_HII`≡0 in the energy phase is a property of the regime, not a theorem** — it survives ~1.5 decades of wind suppression and breaks at 2 | `data/b7_confinement_screen.csv`, `data/b7_regime_trajectory.csv`, `figures/b7_regime.png`, `figures/b7_feedback_compare.png` |
| 8 | ✅ | 2026-08-16 | **C3a IS the classical D-type pressure — the photo-only limit is exact.** No solver run; the shipped `get_phii_c3c` driven through the thin-shell momentum equation. Algebra gates exact to machine precision (Strömgren anchor 2.2e-16; the `mu_convert/mu_ion_shell` prefactor **is** the 2.2 particles per H nucleus, so `P_C3a(R_St) = n_tot k T = rho_0 c_i^2` — Spitzer's `2nkT` with He). Dynamics: index → **0.57124** vs 4/7 = 0.571429, and deviation from Hosokawa–Inutsuka **0.0000% over `R/R_St ∈ [2,50]`** on all 5 `(n_0, Qi)` combinations, while sitting **8.56%** above Spitzer against the analytic `(4/3)^{2/7}` = 8.55% — it lands on the momentum-equation closure, not the ram-balance one. ⚠️ **G8.4 FAILED as registered** (9.511% vs a 5% bar): I compared a from-rest integration against a closed form whose `t=0` state is `v = sqrt(4/3) c_i`, so the gate measured the startup transient (−9.51% at `R/R_St`=2 → −0.01% at 150, index converging on 4/7). Recorded as failed and **amended** (G8.4′), not reinterpreted; the amendment was checked not to weaken the gate — the mis-normalised control still misses by −20.14% vs analytic −20.17%, and the pinned tests fail on `P_C3a × 1.001`. ⚠️ **Not independent confirmation** — HI is *derived from* the same momentum equation, so once the algebra gates hold the ODE must return HI. The content is in G8.2 (the prefactor could have been 1, 2, or a `mu` confusion; it is the He-correct 2.2); the dynamics are a propagation + sensitivity check. **Consequence:** the shipped docstring's "NOT an O(1) normalisation error" is now externally anchored and **confirmed** — the universal HII-dominated momentum phase is not a prefactor bug, so re-deriving C3a's normalisation is a dead end; what stays open is the `R2^{−3/2}` vs `R2^{−2}` geometry, a model-structure question. Both halves of §3's limiting-case obligation on C3 are now discharged | `data/b8_spitzer_crosscheck.csv`, `harness/spitzer_crosscheck.py`, `test/test_phii_c3c_spitzer.py` |
| 9 | 🟡 | 2026-08-17 | **G9.2 FALSIFIED in momentum, G9.3 discharged — and my own scoping headline retracted.** One B3M run (`--arm b9`, code `2fa8cc9c`, clean tree) reproduced the known trajectory exactly (231 rows, 4 phases, `R2_end` 23.253 vs Batch 5's 23.25) and covered the momentum phase the scope could not. The geometry ratio `sqrt(R2/(3 dR))` crosses 1 at `dR = R2/3`, and the shell goes from thin (`dR/R2` ~1e-3, energy) to **thick** (0.670–1.308, momentum), crossing over *inside* transition. So `frac_ratio>1` = 1.0000 / 1.0000 / **0.3810** / **0.0000** across energy/implicit/transition/momentum. ⛔ **The scope's claim that the correction is one-signed and *raises* `P_HII` 1.75–100× is retracted** — in momentum it **lowers** it 0.51–0.71×, i.e. it moves the *helpful* direction, the opposite of what I reported before momentum was covered. ✅ **What survives:** it is not enough — `P_HII/P_ram` median 6.165 → layer-corrected **3.594**, **34/34 momentum rows still HII-dominated**, so a geometry fix cannot produce the wind-dominated branch and D5 (pressure coupling) stays live for a weaker reason than claimed. 🔍 Unregistered but notable: the layer-corrected ratio is nearly **time-independent** (3.584→3.614 over t 0.405→1.5) while the cavity form *climbs* 5.08→7.16 — the growing dominance is a geometry artifact. **G9.4 CLOSED same day: also FALSIFIED (3.171× vs a 2× bar)** — the analytic thin-layer scaling overestimates the real profile's recombination-equivalent density by up to 3.2×, and in the thin-layer phases the gap **is** exactly the dust sink (`sqrt(recomb/Qi_abs)` = 0.497/0.907 vs measured 0.496/0.906, three decimals). G9.2's momentum verdict survives the recheck with the true ionised thickness (0.505–0.712, 0/17 > 1; the shell is 99.54% ionised there, so the clamped `dR` was benign — now measured, not assumed). ⚠️ **But the profile form supersedes this row's 3.594**: the same rows give **median 1.545 (1.322–1.666), falling with time**, so the momentum excess over unity is ~50%, not ~260%. 🔍 Extrapolating stage 3's `Lw^−0.33` puts inversion at **`Lw ≈ 3.4`** rather than 260 — physical, and `B3MW3`/`B3MW10` already exist. Flagged as a lead needing its own measurement, not a result | `data/b9_geometry_scope.csv`, `data/b9_layer_density.csv`, `harness/geometry_screen.py`, `harness/layer_density_check.py` |
| 10 | ✅ | 2026-08-17 | **G10.2 FALSIFIED — the `Lw` ≈ 3.4 lead is dead, and D5 is back to being the route.** `B3MW3` + `B3MW10` ran clean to `stop_t` 1.5, both reached momentum, and their energy row counts (96/105) reproduce stage 3's exactly. Profile-form momentum medians **1.5451 → 1.3412 → 1.1902** for `Lw` = 1/3/10: G10.1 passes by 0.0088 (1.3412 vs a 1.35 ceiling, 25% above its 1.0752 point prediction), G10.3 (cavity form HII-dominated 100%, both rungs) and G10.4 (monotonic) pass, but **G10.2 fails — 1.1902 against a registered < 1.0**. The failure mechanism was registered in advance and is confirmed: stronger winds **thin the shell** (`dR_ion/R2` ∝ `Lw^−0.3375`), so the geometry correction `sqrt(R2/3dR)` *rises* as `Lw^+0.169` and cancels ~43% of the cavity form's `Lw^−0.3959` decline. Net profile response is only **`Lw^−0.1133`**, not the −0.33 my Batch 9 extrapolation assumed. Revised inversion: cavity ≈ `Lw` 99, profile ≈ `Lw` **46.5** — a ~2× improvement, not the ~26× projected, and **still unphysical**. 🔍 Genuine refinement: `frac(ratio>1)` in momentum is 0.0000/0.0000/**0.1667** because `B3MW10` dips to `dR_ion/R2` = 0.3197, below the `R2/3` break-even — so **the geometry correction's sign is wind-dependent**, which reconciles the Batch 9 scope ("raises") with the Batch 9 verdict ("lowers"): both were partial views of a thickness-dependent sign. ⚠️ G10.5 binds — the rungs sit at 0.98/0.74/0.45 thickness, so the trend is confounded with geometry and −0.1133 is not a clean wind response | `data/b10_wind_profile.csv`, `data/b10_walltimes.csv` |
| 11 | 🟡 | 2026-08-18 | **B11.0 DONE — the audit survives an adversarial pass: 3 CONFIRMED, 1 REVISED, 0 REFUTED. A–D not started.** One fresh B3M run at `ef624195` (495.9 s, 231 rows, all 4 phases, `R2_end` 23.253) reproduces `data/b9_layer_density.csv` to **≤3.3e-06 rel on 116 rows × 15 numeric columns**, so the Batch 9/10 baselines are sound and the seams are not a run artefact. **A CONFIRMED:** every `Qi` consumer in `trinity/` enumerated — no cavity-absorption factor at any of them, and the only attenuation (`get_shellODE.py:120`) starts at `r = R2` with `phi0 = 1`; `f_abs` = 1.0000 on 16/16 transition + 13/17 momentum driving rows, so the claimed budget is ≈`2·Qi`. Sharper than §6b put it: `f_abs` is *by construction* the shell's absorbed fraction and `get_bubbleParams.py:358` credits the **cavity** with that identical sub-budget. **B REVISED:** mismatch confirmed and sized (`P_HII/Pb` median 6.1646 momentum, 4.6218 transition; call-time ordering checked in all four runners) but the **direction is wrong** — feedback into `P_C3a` is **exactly zero on 29/33 driving rows** (`f_abs` already saturated) and **upward** on the other 4, so B is not an upper-bound mechanism, and §6b's "every seam pushes the same way" is corrected in place. **C CONFIRMED to 4 s.f. and understated:** `units-reviewer` cleared the derivation (`shell_structure.py:125-126` *is* the shipped inverse); route-P (invert `P_HII`) / route-Q (replay the forward map) agree to **1.000000000000** on all 33 driving rows; the committed `n_cavity` column gives 57,396.6 Msun by a third route. Re-measured **0.0952 (t=0.4074) → 0.5638 (t=1.5), 57,397 vs 101,805 Msun** vs the audit's 0.095 → 0.564, 57,400 vs 101,800. **New:** `shell_mass/M_avail` = 0.999997–1.000000 on every driving row — the shell already holds 100% of the gas that exists — and winds inject only **54.8 Msun** (0→1.5 Myr, from the run's own feedback columns; `bubble_mass` is *frozen* at 99.643 through momentum and unusable). So `(M_cav + M_shell)/M_avail` = **1.5638**: the model asserts 56% more gas than the run has, and over-subscription starts at the **first driving row** (t = 0.3037, transition), not in momentum. **D CONFIRMED:** `dR_full/R2` 0.6723–1.3078, `dR_ion/R2` 0.6579–1.3076, `dR_ion/dR_full` median 0.9954. **Side-findings S1–S4** (outside the seams, they change committed numbers): S1 `layer_density_check.py:140`'s thin-shell `V_lay` overstates `n_layer_analytic`/`ratio_analytic`/`pdrive_analytic` by 1.342–1.696× in momentum (`V_exact/V_thin` 1.802–2.878) — G9.2/G9.4/G10.1–4 verdicts all survive, and so do Batch 10's `Lw^−0.1133` fit and `Lw ≈ 46.5` inversion (both fitted to `pdrive_profile`, which never uses `V_lay` — re-derived from the published medians to check), but Batch 10's "flat to 1–3% within each rung" is withdrawn and the affected columns are re-fit in **B11.F**; S2 `layer_density_check.py:154`'s falsy-zero filter drops `P_HII == 0.0` rows (0/17 in momentum, so no published number affected); S3 `run_momentum_phase.py:888-896` never recomputes `P_HII` (no effect here — proved by the route agreement); S4 `b9_walltimes.csv` gains a real B3M timing, 495.9 s vs the mtime-derived ~590 s. **External:** all four seams are absent by construction from Geen et al.'s two-equation closure — registered as **B11.G** | `data/b11_mass_ledger.csv`, `harness/mass_ledger_check.py`, `data/b9_walltimes.csv` |
| 5-s3 | ✅ | 2026-08-13 | **Wind ladder DONE — Lancaster resolved in transition, open in momentum.** First ladder (`simple_cluster`: SC/SW3/SW10) **VOID** — all three terminate at `stop_t` still in implicit, so `t_cross = never` is not evidence about winds; the crossover is structurally floored at the transition handover (`ratio@entry` < 1 on every config). Re-run on B3M (`B3MW01/1/3/10`), all four valid. **Transition: (a) PASSES** — confined fraction 8.8% → 23.8% → 28.9% → 38.8%, lag/`t_entry` +1.5% → +37.2%, and `ratio@entry` = 0.7144/0.1227/0.0553/0.0235 vs the **pre-registered** 0.68/0.12/0.054/0.022 (exponent **−0.743** vs −0.74, errors 2–7%) — a Weaver-derived prediction holding out of its fitted regime. **Momentum: OPEN** — 100% HII-dominated on all four; `P_C3a/P_ram ∝ Lw^−0.33` ⇒ inversion needs `Lw ≈ 260`. **Not** an O(1) normalisation error (same normalisation predicts transition to 7%): it is the `R2^−3/2` geometry. Mechanism: energy duration **identical** (0.0030 Myr) and transition duration wind-independent; only implicit moves (`Lw^−0.388`), which is why `t_cross` *falls* with wind. ⚠️ Retracted mid-run claim that energy row counts (69/87/96/105) meant longer energy phases — that is timestep refinement, not duration | `data/b5s3_ladder_lag.csv`, `data/b5s3_ladder_regime.csv`, `data/b5s3_ladder_screen.csv`, `harness/lag_vs_handover.py` |

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

⚠️ **2026-08-13 (audit).** The SHA column means *code that produced the runs*, which is **not** the
same as the C-6 stamp inside each file (the stamp is the harness tree's SHA at write time, so a CSV
regenerated later stamps a newer SHA than the runs it summarises). Where the two differ, the stamp
is authoritative about *when it was written* and this column about *what it describes*. Two legacy
CSVs (`phii_identity_evidence.csv`, `roundtrip_ulp.csv`) carry **no C-6 stamp at all** — by this
plan's own "no stamp, no trust" rule they are the least trustworthy artifacts here, and both predate
the rule being enforced.
| file | producer | batch | stamp SHA |
|---|---|---|---|
| `data/phii_identity_evidence.csv` | (evidence phase) | pre | `6d84b1e` |
| `data/roundtrip_ulp.csv` | `harness/roundtrip_ulp.py` | pre | `6d84b1e` |
| `data/b1_bitidentity_ww.csv` | `harness/compare_bitidentical.py` | 1 | `088a8d6` |
| `data/b4a_ledger.csv` | `harness/compare_trajectories.py` | 4a | `088a8d6`+cap-removed |
| `data/b4a_identity_grid.csv` | `harness/harvest_identity.py` | 4a | `088a8d6`+cap-removed |
| `data/b4a_walltimes.csv` | `harness/run_batch.py` | 4a | `088a8d6`+cap-removed |
| `harness/b4a_cap_removal.patch` | the exact 4a code change (apply to `088a8d6` to reproduce) | 4a | `088a8d6` |
| `harness/b3_c1_momentum_max.patch` | the exact C1 arm diff — **momentum sites only** (`run_momentum_phase.py:265,445`); §3's "5-site" wording predates D1's ruling that the transition `max` is deliberate | 3 | `41511ac` |
| `data/b3b_coupling_regression.csv` | `harness/coupling_regression.py` | 3b | `3a38a87`+dirty |
| `data/b3_c1_ledger.csv` | `harness/compare_trajectories.py` | 3 | `41511ac`+C1 patch |
| `data/b5_c3_screen.csv` | `harness/c3_offline_screen.py` | 5 (stage 1) | evaluated on b1 runs; no arm |
| `data/b5_c3c_regime.csv` | `harness/c3_offline_screen.py --regime-out` | 5 (stage 1b) | evaluated on b1 runs; no arm |
| `data/b5_c3c_seams.csv` | `harness/c3_offline_screen.py --seams-out` | 5 (stage 1b) | seam/switch continuity; evaluated on b1 runs |
| `data/b5s2_c3c_ledger.csv` | `harness/compare_trajectories.py` | 5 (stage 2) | arm vs b1 baselines, matched-t |
| `data/b5s2_c3c_arm_regime.csv` | `harness/c3_offline_screen.py --regime-out` | 5 (stage 2) | the arm's OWN regime structure |
| `harness/b5s2_c3c.patch` | the exact C3c arm diff (apply to `c01626e`) | 5 (stage 2) | helper + 6 call sites; no `P_drive` edits |
| `data/b0_identity_grid.csv` | `harness/harvest_identity.py` | 0 | runs @ `6b55657` |
| `data/b0_trajectories.csv` | `harness/harvest_identity.py` | 0 | runs @ `6b55657` |
| `data/b0_walltimes.csv` | `harness/run_batch.py` | 0 | runs @ `6b55657` |
| `data/b1_bitidentity.csv` | `harness/compare_bitidentical.py` | 1 | b0 `6b55657` vs b1 `bb302e0` |
| `data/b1_capmap.csv` | `harness/harvest_identity.py` | 1 | runs @ `bb302e0` |
| `data/b8_spitzer_crosscheck.csv` | `harness/spitzer_crosscheck.py` | 8 | no run — closed-form on the shipped helper |
| `data/b11_mass_ledger.csv` | `harness/mass_ledger_check.py` | 11 (B11.0) | B3M re-run @ `ef624195`; replays `shell_structure_pure`, does not integrate |

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
  separate pressure. Three consequences, all recorded above. (1) C1 was briefly marked ⛔ on the
  strength of the intent ruling alone; **that verdict is retracted the same day** — it was declared
  without a single run, which this workstream's own bar forbids. C1 is reset to ⬜ pending Batch 3.
  The ruling still makes the circularity `P_ram → Pb → shell_n0 → n_IF_Str → P_HII` the more likely
  defect than the sum, but "more likely" is a hypothesis, not a verdict. (2) New §3b proves, with N = 803 rows over
  8.8 decades *(⚠️ those figures were corrected on 2026-08-13 — see the audit entry below; the pool included two aborted runs. Corrected: N=788, slopes −2.348/+1.036, conclusion unchanged)*, that **the cap is only the last link**: `ΔV ∝ shell_n0^-2.126` and
  `n_IF_Str ∝ ΔV^-1/2` give `n_IF_Str ∝ shell_n0^+1.039` (r = 0.993) *before* the cap applies, and
  Batch 4a's uncapped runs still never put `P_HII` below `Pb`. The intervention point is the shell
  ODE's inner boundary condition at `shell_structure.py:124-126`, not the cap at `:253`.
  (3) **Batch 4b is deprioritised** — replacing the guard cannot decouple anything, because the
  guard is not what couples. Batch 5 (C3) is promoted to next, and its first stage is an *offline*
  screen of C3a/C3b against committed snapshots, since both are closed-form in already-stored
  quantities and the likely failure mode is magnitude rather than stability. C3a measured offline on
  B3M momentum rows: 235 → 47 cm⁻³, P/k 5.2e6 → 1.0e6 K cm⁻³, decoupled, but ≈5–7× `P_ram`
  (figures corrected 2026-08-13 by audit — see §3b).

- **2026-08-13 (independent audit — two CRITICAL, eleven MAJOR findings; all fixed here)** — An
  adversarial audit was commissioned specifically to assume this file over-claims. It did, in ways
  worth recording rather than quietly patching:
  1. **CRITICAL — the cap's line number was wrong in all 7 citations.** Batch 1's own diagnostic
     commit inserted two lines above it, moving the cap from `:251` to `:253`, and no citation was
     updated. C2a's instruction "delete `shell_structure.py:251`" would have deleted the Batch-1
     diagnostic and left the cap in place. (The committed 4a patch was correct; only prose was wrong.)
  2. **CRITICAL — the retracted "removal lowers the drive" prediction still stood, unmarked, inside
     the block headed "read this paragraph with the corrections below".** Retracting a claim in §6
     while leaving it live in §1 is not a retraction. Struck in place.
  3. **The blow-up maximum was wrong by 2.3×.** `b1_capmap.csv` was never regenerated after the WW
     and SC b1 arms landed, so 3.33× (B3M) was quoted as the matrix maximum in four places. It is
     **7.786×** (WW momentum). The C2a kill bar was 1e2, so no verdict changes — but the plan had
     explicitly pre-registered "run b1 WW before Batch 4a and re-check the kill bar against it", the
     WW run was made, and the re-check was silently skipped. Done now.
  4. **§3b's regression pooled two aborted runs**, one duplicating ~110 rows of its own completed
     re-run (C-7), and cited no artifact. Recomputed over the four *complete* runs and committed as
     `data/b3b_coupling_regression.csv`: pooled slopes −2.348 / +1.036 (were −2.126 / +1.039), with
     a per-config spread of −2.02…−2.72 and +0.996…+1.096 now reported as the honest error bar. The
     conclusion is unchanged and if anything stronger: **frac(uncapped `P_HII` < `Pb`) = 0.0000 in
     every config.**
  5. **The C3a screen was internally inconsistent by 2.2×** — its pressures omitted the
     `mu_convert/mu_ion_shell` factor the code applies while its `P_ram` ratio included it, so the
     "physically reasonable magnitudes" verdict rested on the low number. Corrected to
     P/k = 5.2e6 → 1.0e6 K cm⁻³.
  6. **"Faster than baseline" was false** (PRB 764 s vs 713 s, 7% slower); **"100% of momentum rows"**
     was false (WW is 26/27); **"4/4 configs"** hid that Batch 4a ran **4 of the pre-registered
     core-6**, omitting WW — the very config with the largest blow-up.
  7. **The 4a raw outputs were deleted with a scratch worktree and the docs never said so.** Now
     disclosed, with the three claims that are consequently unverifiable named explicitly.
  8. Also fixed: §7 still listed D1 as open a day after it was answered; README §5/§7.3 asserted
     "only 1a/1b are safe" and "`_yesPHII`/`_noPHII` differ only via momentum", both superseded by
     D-ramp and mutually contradictory within one file; Batch 2's rationale still rested on the
     retracted cap-slack premise; the artifact index's SHA column conflated "stamp" with "code that
     produced the runs"; two legacy CSVs carry no C-6 stamp at all.

  **Process lesson, recorded because it recurred:** every one of these is a *bookkeeping* failure on
  top of measurements that survived recomputation. The physics held up under adversarial re-derivation;
  the layer of headline numbers on top of it did not. Retraction discipline in particular failed twice
  in the same way — correcting a claim where it was *discovered* rather than everywhere it was
  *written*. Future visits: when retracting, grep the retracted phrasing across the whole workstream
  and mark every copy, and regenerate derived CSVs whenever a new arm lands rather than at write time.

- **2026-08-13 (Batch 3 — C1 measured, and the retraction vindicated)** — C1 (momentum-only
  `max(P_HII, P_ram)`) run against matching b1 baselines on four configs: B1M, B2M, B3M and WW
  (weak winds). **All within the 5% bar, no fate changes: 0.000% / 1.24% / 4.00% / 1.29%.** The
  pre-registered control worked — B1M never reaches the momentum phase, so C1 had to be inert there,
  and it is to 0.000% at matched `t`. Halving the momentum drive costs ≤4% in final radius because
  momentum is only 12–15% of these runs; weak winds is not the worst case, the densest bench is.
  **Verdict: C1 is safe but wrong-target** — with `P_HII ≡ P_ram`, `max(P_HII, P_ram) = P_ram`, so it
  deletes the photoionised channel instead of decoupling it (against D2), and D1 rules the sum
  intended. Superseded by C3; kept as the measured price of the double-count, ≤4% ΔR2, which is the
  bar any C3 formulation must justify clearing.
  Two methodological by-products, both now baked into the tooling: (a) the B1M control quantified a
  **cross-worktree noise floor** — physical keys agree to machine precision until t ≈ 0.8 Myr then
  drift to at most 2.9e-14 in R2, seeded by `Lmech_SN`, which is `Lmech_total − Lmech_W` and thus
  exactly zero pre-SN (the stored ~1e-18 is a ~1e-26-relative cancellation remnant). That is 13
  orders below Batch 4a's 15–28%, so 4a's conclusions are unaffected. (b) `compare_bitidentical.py`
  gained that floor plus array-aware and per-key-scale comparison, because strict bit-identity is the
  right gate for a *diagnostic* change and the wrong one for a cross-worktree comparison — it was
  reporting "100% difference" on a quantity that is identically zero.

- **2026-08-13 (Batch 5 stage 1 — the offline C3 screen; C3b dies, C3a advances)** — Both decoupling
  candidates screened without running the solver, on five complete b1 runs. **C3b is rejected on the
  acceptance floor this plan pre-registered before any C3 existed**: `n = n_cloud(R2)` contains no
  `Qi` term, so it cannot reproduce the wind-only limit — turn the cluster off and its `P_HII` does
  not move. Its other failures (it is the *neutral* gas ahead of the shell, and it steps from `nCore`
  to `nISM` at `rCloud`, swinging four decades on a geometric boundary) are corroborating, not the
  reason. **C3a passes**: causally decoupled, correct `Qi → 0` limit, sensible ionised densities. Its
  cost is that it predicts a photoionisation-dominated momentum phase in every config (3.5–7.6×
  `P_ram`, never crossing), which is falsifiable but large.
  Two things worth recording beyond the verdict. (1) **The decoupling metric nearly fooled me.** C3a
  scores slope ≈ 0.7–1.1 against `Pb` in several phases despite never reading `Pb` — because both
  decline together along a trajectory. Shared time-dependence is not dependence; the honest
  discriminator was the structural question "does this depend on `Qi` at all", not the regression.
  Stock's exact 1.0000/1.0000 is the only slope in the table that means what it looks like.
  (2) **A harness bug was caught in smoke-testing and is worth remembering**: `read_param` leaves
  `rCloud = 0` because it is derived during cloud init, so `get_density_profile` treated every radius
  as outside the cloud and C3b silently reported the ISM density *everywhere* — including the energy
  phase, where R2 is deep inside the cloud. The screen now overlays the run's `metadata.json`
  constants and refuses to report C3b at all if `rCloud` is unavailable. Had that gone unnoticed,
  C3b would have been rejected for the wrong reason and the record would have looked identical.

- **2026-08-13 (C3c designed, then both screened together)** — The design pass (§3c) turned up
  something better than the brief: "keep the skin, decouple its inner BC" is **ill-posed**, because
  every decoupled closure either re-couples (jump condition → shell density; mass closure → shell
  structure) or lands back on the C3a scaling with a *higher* pressure (any independent-thickness
  skin has smaller ΔV, and Strömgren pressure ∝ ΔV^(−1/2) — the cavity-filling C3a is the *minimum*).
  The only remaining closure, pressure equilibrium with the confinement, **is the current code** —
  which is what the cap has been measuring all along. So the physical content of C3c is a **regime
  switch**: the ionized layer transmits (contributes nothing independent) while `P_C3a ≤ P_conf`, and
  drives at `P_C3a` once confinement cannot hold. Joint screen (`b5_c3c_regime.csv`, same five runs,
  still no solver): implicit phase **exactly** untouched (ratio 1..1..1 — the falsifiable control);
  energy ratios 0.30–1 which *is* the D-ramp fix (the ramp finally biting, 1/3.3 ≈ 0.30); crossover
  inside transition in all four configs that reach it (0.16–0.67 Myr); momentum 100% HII-dominated at
  2.4–4.3× stock drive. One correction to stage 1's wording, marked in §3c: the coevolution crossover
  *does* appear — vs the confining pressure in transition, not vs `P_ram` within momentum. C3c
  supersedes bare C3a; stage 2 is the run arm, and its open risks are fate flips and the integrator's
  tolerance of the handover kink, both only answerable by running it.

- **2026-08-13 (§3c.1 — continuity at the regime switch and the phase seams)** — Raised by the
  maintainer; measured offline on the same five runs (`data/b5_c3c_seams.csv`). Result: **C3c is as
  smooth as or smoother than stock at every seam.** Energy→implicit 0.89–0.92 vs stock's 0.81–0.84;
  transition→momentum 0.995–0.999, continuous *by construction* because `P_C3a` does not know the
  phase label (stock's `P_HII` is redefined from `Pb`-slaved to `P_ram`-slaved at that seam). The
  regime switch itself is nearly invisible (0.86–0.99 on the same rows where stock, which has no
  switch, moves 0.75–0.83): `max` is C0 — the branches are equal at the crossover, so only the
  derivative kinks. One honest subtlety recorded: at implicit→transition, stock *looks* marginally
  smoother in WW (0.586 vs 0.526) — but that is stock's instantaneous `+P_ram` switch-on partially
  masking the real decline of `Pb`; the discontinuity is stock's, and C3c mediates it through the
  `max`. Pre-registered remedy if the stage-2 arm stumbles at `t_cross`: solver event via the
  existing `phase_events.py` machinery; smooth-max is fallback only, since its width parameter is a
  new magic number and needs a maintainer ruling.

- **2026-08-13 (§3c momentum branch clarified; stage 3 regime map proposed)** — A maintainer question
  ("can we test schemes where `Pb` dominates over `P_HII`?") exposed an ambiguity in §3c: the
  per-phase drive table gave momentum an unconditional `P_C3a + P_ram`, while the branch rule says
  the confined ionized layer contributes nothing independent. Reconciled in place: **the branch rule
  is primary in every phase** — strong winds or faded `Qi` in momentum give `P_ram` alone. The
  ambiguity was invisible in current data (momentum is 100% HII-dominated in all screened configs),
  which is exactly why it needed writing down before a config reached the other corner. Stage 3
  (regime map) added to Batch 5: strong-wind rungs (`FB_thermCoeffWind` 3/10), the low-`Qi` corner,
  and `stop_t 15` runs to catch the post-SN `Qi` fade — where C3c predicts a possible **second
  crossover back to confinement**, a falsifiable prediction stock cannot express at all (its cap
  makes "`Pb` ≥ `P_HII`" an identity rather than a measurable regime). Also noted: the `Pb`-dominated
  regime is not hypothetical — it already holds through 100% of energy/implicit rows in every
  screened config, and through PRB's entire run.

- **2026-08-13 (Batch 5 stage 2 — the C3c arm runs, and the screen is vindicated)** — Five configs,
  zero numerical distress, **no fate changes**, ΔR2 12.8–20.5% (pre-registered as expected). The
  falsifiable null passed **exactly**: `P_HII` = 0 on 0 of 330 implicit rows and `P_drive` == `Pb` to
  machine precision, so the confined branch is wired in correctly; in the energy phase `P_drive`
  equals the *ramped* pressure on 86/87, 75/76, 64/65 rows, which is the D-ramp fix visible directly.
  The `t_cross` kink did not trouble the integrator on any config — §3c.1's event-detection remedy
  stays registered but is unneeded on this evidence.
  **The methodological result is the one to remember:** the offline screen, computed on *stock*
  trajectories, predicted the self-consistent arm's regime structure to the printed digit on B3M,
  B2M and PRB (`t_cross` 0.301207 and 0.449094 exactly; PRB "never" exactly), 3% off on B1M, and 28%
  early on WW — the one config whose trajectory changed enough to move its own crossover. A cheap
  screen on stock trajectories is therefore a trustworthy filter for this class of change, which is
  what makes stage 3's regime map affordable. Implementation fidelity was checked independently by
  re-running the screen on the arm's own output: drive ratio 1.000 in every phase of every config.
  One correction to my own earlier framing, recorded because it nearly went out wrong: I had called
  the implicit phase a "0% ΔR2 null". That is sloppy — implicit *inherits* the energy-phase offset,
  so its ΔR2 is non-zero by construction (1.1–3.9% measured). The null is about the **drive**, and
  tested that way it passes exactly. Only D3 (WW's 16%-earlier collapse) and stage 3 remain before
  C3c can carry a physics verdict.

- **2026-08-13 (Batch 5 stage 3 — the wind ladder, a void experiment, and a split verdict)** — The
  stage-3 discriminator as originally written was **unsafe**, and the first ladder proved it rather
  than answering it. On `simple_cluster`, SC/SW3/SW10 all terminate at `stop_t` **still in the
  implicit phase**, so all three report `t_cross = never`. Read at face value that is a triumphant
  confirmation of Lancaster; it is an artifact. The C3c crossover is structurally floored at the
  energy→transition handover (`ratio@entry` is 0.12–0.71 across every complete run, always < 1), so
  a cloud that never reaches transition cannot cross at **any** wind strength. The lesson is now a
  tool: `harness/lag_vs_handover.py` reports such runs as **VOID**, not "never crossed", and the
  void rungs stay in the MATRIX commented with their reason so this is not re-derived expensively.
  Re-run on B3M — which spends 42 rows in transition and 34 in momentum — the ladder splits:
  **transition passes (a)**, with the confined fraction growing 8.8% → 38.8% over two decades of
  wind, and `ratio@entry` = 0.7144/0.1227/0.0553/0.0235 against the **pre-registered**
  0.68/0.12/0.054/0.022 (exponent −0.743 vs −0.74, per-rung error 2–7%). That prediction was
  committed (`70f8711`) while the runs were in flight and flagged in-doc as an extrapolation out of
  its fitted regime; it held anyway, which is the strongest evidence in this workstream that the C3a
  normalisation is *right*. **Momentum stays open**: 100% HII-dominated on all four rungs, with
  `P_C3a/P_ram ∝ Lw^−0.33` ⇒ inversion at `Lw ≈ 260`. Both of branch (b)'s *triggers* fired while
  (b)'s *conclusion* was false, so the registered dichotomy is recorded as **mis-specified** rather
  than resolved — the fix is not a smaller prefactor but the `R2^−3/2` cavity geometry itself.
  The mechanism only became visible from phase **durations**: energy lasts *exactly* 0.0030 Myr at
  every wind strength and transition is wind-independent; only implicit moves (`Lw^−0.388`), which
  is why `t_cross` falls with wind while confinement strengthens. ⚠️ **Retraction:** mid-run I read
  energy-phase *row counts* (69/87/96/105) as longer energy phases. Wrong — that is timestep
  refinement under stiffer winds. Row count is not duration, and here they point opposite ways.
  With **D3 and D4 answered** (§7), Batch 6 is unblocked.

- **2026-08-14 (C3c landed; two loose ends recorded, not closed)** — `c43a50e` merged to `main` in
  `186cc5a` (PR #738): six `P_HII` call sites across `phase1_energy/run_energy_phase.py`,
  `phase1b_energy_implicit/run_energy_implicit_phase.py`, `phase1c_transition/run_transition_phase.py`
  and `phase2_momentum/run_momentum_phase.py` now call `get_bubbleParams.get_phii_c3c`. This entry
  exists because the landing was *not* recorded anywhere at the time — DOC_STATUS, this doc's Status
  line and `docs/dev/README.md` all still described C3c as a candidate under evaluation while it was
  the production model. Fixed 2026-08-14 in all three.
  **What the landing proved, on the suite rather than on the harness:** §3c's third consequence —
  "**D-ramp is fixed as a side effect.** In the energy phase `P_C3a ≪ Pb`, so the `max` selects the
  ODE's own (ramped) bubble pressure" — is what the suite is now reporting, in the form of moved
  goldens. ⚠️ **Do not cite §3 item 3's struck sentence here.** That sentence ("removing the cap
  drops early driving pressure by up to 3×") was retracted 2026-08-13 as backwards, and correctly so:
  it was about **C1** (removing the cap), which raises `P_HII` *above* `Pb` and moves ΔR2 **up**. C3c
  is a different intervention — it does not unclamp the density, it replaces the quantity — and it
  lowers the energy-phase drive by zeroing the channel entirely. Same direction of golden movement,
  opposite mechanism; the struck sentence is still struck. Three default-suite tests are red on
  `main`: `test_run_smoke.py` (`R2` −1.09%), `test_phase_boundary.py` (`cool_beta` −1.10%), and
  `test_mu_audit_drift.py::test_phase1_all_eleven_sites_refined_and_no_original_remains`, which is
  pure bookkeeping — the refined `mu_convert/mu_ion_shell` factor count in the five phase files went
  11 → 5 because six of the sites moved into the helper, where the factor still appears once. No
  original `* 2.0 *` operation came back; the n-consistency invariant is intact.
  **The mechanism, stated once so it is not re-derived:** in phase 1a the RHS drive is
  `P_drive = max(press_bubble, P_HII)` (`energy_phase_ODEs.py:256` @ `c43a50e`), where
  `press_bubble` is the `dt_switchon`-ramped pressure and the old `P_HII` was `params['Pb']`
  relabelled — un-ramped, and frozen per segment. So the `max` selected the un-ramped floor and the
  ramp never reached `vd` at all; it acted only through `Ed` and `L_leak`. C3c returns exactly `0.0`
  on the confined branch, so `P_drive` is the ramped pressure alone and the ramp throttles the shell
  for the first time. That is D-ramp being fixed, and it is why the shift is 1.1% rather than 0.
  **Consequence for a sibling workstream:** `switchon-successor/` measured `dt_switchon` entirely in
  the pre-C3c regime. Its algebra (D1, D4 — `PdV/Lmech = 2(v2/v_wind)/(R1/R2)^2`, `E0` absent) is in
  `Ed` and survives untouched; its ablation fates, cost bounds and Weaver-N1 comparisons are not
  quotable until re-measured. The 50-line rationale block that Batch 5 wrote into
  `get_bubbleParams.get_effective_bubble_pressure` carries those figures, so it now carries a dated
  pre-C3c provenance note as well.

- **2026-08-18 (Batch 11, B11.0 — the audit was attacked, and it held)** — The maintainer ruled that
  §6b "must not be assumed correct", so B11.0 tried to kill each seam rather than confirm it.
  Outcome: **A, C, D CONFIRMED; B REVISED; none REFUTED.** The single most load-bearing check was on
  seam C, flagged in advance as the highest-risk item because it is a units-sensitive derived number.
  Four independent attacks, all of which it survived: (1) the `units-reviewer` agent found the
  inversion exact and, decisively, that `shell_structure.py:125-126` **is** the shipped inverse of
  the same map — the audit used the code's own convention rather than inventing one; (2) a new
  harness computes the cavity density both by inverting `P_HII` and by replaying the forward
  `get_phii_c3c` map through `shell_structure_pure`, and the two agree to **1.000000000000** on all
  33 driving rows, which a units or algebra error could not have survived; (3) the committed
  `n_cavity` column — built from `Qi_abs`, never from `P_HII` — reproduces 57,396.6 Msun by a third
  route; (4) an order-of-magnitude sanity check on the implied `Qi`. The claim reproduced to four
  significant figures.
  **Where the audit was wrong: seam B's direction.** §6b listed B under "every seam pushes the same
  way, `P_C3a` is an upper bound". Measured, `f_abs` is already 1.0000 on 29 of 33 driving rows, so
  raising the shell's inner boundary pressure cannot change `Qi_abs` at all — the feedback is
  *exactly zero* there — and on the remaining 4 rows a denser inner edge absorbs *more*, pushing
  `P_C3a` **up**. B is a real inconsistency (thickness, dust column, gravity sampling) but it
  inflates nothing. Both §6b's seam-B entry and its summary paragraph are corrected in place.
  This also re-scoped **B11.B**, which as registered would have replaced the *defensible* side of the
  mismatch: Geen et al. close the ionised gas with `P_w = n_i c_i² m_H/X` at the wind-bubble edge,
  which is exactly `nShell0 ∝ P_ram`, so the shipped boundary condition is the standard one.
  **Where the audit was too kind to itself: seam C is worse than it said.** §6b framed it as an
  either/or — "either the cavity is filled and the shell should be ~2× lighter, or it is
  supply-limited". Measured, `shell_mass` equals **100.0000%** of all the gas the run has
  (0.999997–1.000000 on every driving row: 100,000 Msun of cloud gas plus 1,805 Msun of ambient swept
  beyond `rCloud` = 4.999 pc), and the wind injects **54.8 Msun** over the whole run — 1/1047 of
  `M_cav`. The filled-cavity limb needs `(M_cav + M_shell)/M_avail` = **1.5638**, i.e. 56% more gas
  than exists, so it is not available and the fork collapses to the supply-limited branch.
  Over-subscription also starts at the **first driving row** (t = 0.3037, transition), not in
  momentum as §6b implied. ⚠️ Recorded because it was nearly used as evidence: the snapshot's
  `bubble_mass` looks like the natural wind-mass cross-check, but the momentum phase never
  recomputes it — it is frozen at 99.643 Msun on all 34 momentum rows — so the feedback-column
  integral was used instead. The `units-reviewer` pass caught this before it reached a conclusion.
  **Side-findings that change committed numbers (S1–S4 in §Batch 11).** The one that matters is S1:
  `layer_density_check.py:140` uses a *thin-shell* layer volume in the momentum phase, where
  `dR_ion/R2 ≈ 0.98` and the shell is emphatically not thin. `V_exact/V_thin` = 1.802–2.878, so
  everything built on `n_layer_analytic` is overstated by 1.34–1.70×. Every affected *verdict*
  survives — G9.2 moves further below 1, G9.4's worst case sits in the thin energy phase and is
  unchanged at 3.171, and G10.1–G10.4 never used `V_lay` — but Batch 10's "flat to ~1–3% within each
  rung" is an artefact of the approximation (the exact form spans ±12%) and is withdrawn.
  ⚠️ Recorded because I got it wrong first: I initially withdrew Batch 10's `Lw^−0.1133` exponent and
  `Lw ≈ 46.5` inversion along with it, on the assumption they inherited the bias. They do not — both
  are fitted to `pdrive_profile`, which never touches `V_lay`. Re-deriving them from the published
  profile medians (1.5451 / 1.3412 / 1.1902) returns 0.1133 and 46.5 exactly, so they stand.
  **New work registered.** B11.F (re-fit Batch 9/10 on the exact layer volume, no run needed) and
  **B11.G** — the useful by-product of reading Geen et al., "When H II Regions are Complicated": their
  two-equation closure (photoionisation equilibrium over `r_i³ − r_w³`, plus wind/photoionised
  pressure balance at `r_w`) has **all four seams absent by construction**. One `n_i`, one `r_i`; the
  ionised gas consumes `Q_H` once, the wind bubble holds no photoionised gas, and the pressure
  balance *is* the boundary condition. That turns D5 from a from-scratch design into a comparison
  against published algebra. Caveat kept explicit: parts of that paper assume `w = 2` while B3M is
  uniform, so only the profile-independent equations transfer.
  Artifacts: `data/b11_mass_ledger.csv`, `harness/mass_ledger_check.py`; `data/b9_walltimes.csv` now
  carries a real B3M timing (495.9 s / 231 snapshots) in place of the `skipped` row Batch 9's
  `--root` bug left. No `trinity/` source was touched — B11 is measurement and verification only.
