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

**B11.A–D DONE (2026-08-18) — the seams are quantified, and the "upper bound" story is down to one
of them.** ⛔ **G11.A2 refutes §6b seam A's consequence clause**: the photon-conserving fixed point
gives `P_C3a_fixedpoint/P_C3a_shipped` = **1.0000–1.1778 with 0 of 33 rows below 1**, where §6b
predicted < 1 — repairing the double-spend *raises* `P_C3a`. And **G11.A1** shows the fixed point is
**degenerate** (unique root `x = 1` on every driving row: cavity takes all photons, shell left
neutral), so **C3a cannot be made photon-conserving at all** without a second equation. **G11.B1/B2**:
both falsifiers of B11.0's seam-B revision failed to fire; the inconsistency is large (`shell_n0`
×4.70/×6.17, layer thins 79–83%, dust fraction 0.620→0.455 / 0.607→0.395 — so seam B and G9.4's dust
are not independent). **G11.C1**: the cavity is **not** rate-limited (supply/required 1.32–2.13 on
100% of rows), which closes §6b's "supply-limited" limb while B11.0's "no reservoir" stands.
**G11.C2**: control passes at 0.871% (2% bar), and debiting the shell by `M_cav(t)` is worth
**+8.55%…+9.22%** in `R2` at t=1.5 — ⚠️ *below* my pre-registered 10–30%, recorded as a miss.
**Net: §6b's "every seam pushes the same way" list loses both A and B; only seam C and G9.4's dust
bound `P_C3a` from above.** The hold's stated release criteria are met; the release is the
maintainer's call, and the numbers move D5's question from "C3c-switch vs C3a-raw" to "C3a at all".
Evidence: `data/b11_photon_ledger.csv`, `data/b11_mass_dynamics.csv`.

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

**C-0 EXTERNAL-DOCUMENT CARVE-OUT (added 2026-08-18, maintainer ruling).** The rule above was
written against *our own* sprawl and it wrongly caught a different thing: a document we did **not**
author. A third-party review, a paper summary, or any assessment handed to the workstream from
outside is an **input**, in the same category as a `data/*.csv` — not a workstream note. Such a file
may live in the workstream folder, under all of the following conditions, and a file failing any of
them is a §0 violation and must be folded into this doc or removed:
1. **Not authored here, and kept verbatim.** We do not rewrite its body into our voice. Corrections
   go in a clearly separated section, never by editing its claims in place. (If we would need to
   rewrite it, it is our note and §0 applies — fold it in.)
2. **Provenance block** naming the author, the date, and its standing, directly under the Status
   line, stating explicitly that it is not this workstream's own measurement.
3. **A cross-check section** at the top separating what we independently verified, what the primary
   sources corroborate, and what our measurements contradict — written by us, dated.
4. **Indexed in §8.3** like any other artifact.
5. **Never load-bearing.** No §8 ledger verdict and no gate may cite it as evidence. It can motivate
   work and supply references; the evidence must be re-derived here. Cite it for attribution only.

Currently exercised by exactly one file: `LITERATURE_ASSESSMENT.md`.

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

### Batch 5 — C3: the advanced method — Status: ✅ **ALL STAGES DONE** — 1 (offline screen: C3b rejected, C3a advances), 1b (C3c designed), 2 (run arm clean on 5/5), 3 (wind ladder: transition passes, momentum open). C3c then landed in `c43a50e`; see the Status block at the top of this file.

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

**Owed items now discharged (2026-08-14, on `hotfix/CI-check`).** Goldens re-baselined under D4
with the G3.4 before/after table committed at `data/g34_golden_rebaseline.csv` — mechanism named in
its header, reproduce commands included, every `rel_change` recomputed from its own two columns:

| test | before | after |
|---|---|---|
| `test_run_smoke.py` | `R2`/`v2`/`Eb` 0.25955976 / 49.226112 / 662533.97 | 0.25672223 / 48.944359 / 657558.38 |
| `test_phase_boundary.py` | (0.888197, −0.046294) ×2 | (0.878396, −0.038973) ×2 |
| `test_betadelta_hybr_stress.py` | + (0.845829, −0.145668) ×2 | + (0.842071, −0.151456) ×2 |

`test_scheme_screen` needed no change. `test_mu_audit_drift` was **fixed, not re-baselined**: its 11
refined sites are now 5 inline + 6 reached through `get_phii_c3c`, and it asserts that accounting.
Full `pytest` on the merged tree: **green**. Two measurements worth keeping — the `stop_t=0.004` and
`stop_t=0.008` runs agree to all printed digits on the rows they share (so `test_phase_boundary` and
`test_betadelta_hybr_stress` may legitimately carry one pair), and the headroom was checked rather
than assumed (`cool_beta` spans 3.4e-7 across CI's four Pythons against `abs=2e-3`; smoke `R2` spans
1.6e-9 relative against `rel=1e-6`).

⚠️ **This block's own caveat held:** `test_run_smoke` was **not** on D4's list. It was re-baselined
on the maintainer's direct instruction, 2026-08-14 — recorded rather than left to look covered.

**Still owed:** CHANGELOG entry; fold-back notes to `feature/threeway-pt2` and
`feature/low-winds-regime`; the twelve non-B3M configs remain unretested against `main` (below).

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
under D4 with a table of before/after — **done, see above**.

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

### Batch 11 — verify, then quantify, the four driving-branch seams — Status: 🟡 **B11.0 + B11.A–D DONE (2026-08-18); B11.E/F/G open**

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

*External corroboration — four sources, and C3a is the only outlier.* Geen et al. 2019
`wind:photoequilibrium` and Geen & de Koter 2022 `eqn:photoionisation_equilibrium_uniform` both write
`Q_H = (4π/3)·n_i²·(r_i³ − r_w³)·α_B`; Lancaster Paper I `eq:ionreceq2` writes the identical
condition and adds that "the WBB enhances `ρ_i` … due to the presence of `R_w` in the denominator";
and **trinity itself already does it** at `shell_structure.py:243` (`_vol_ion = R_IF**3 -
rShell0**3`). All four exclude the wind bubble, and the photoionised gas lives between `r_w` and
`r_i`. C3a balances over `(4/3)πR2³` — in trinity's geometry `(4/3)π r_w³`, *exactly the volume all
four exclude*. So the cavity mass C3a implies is not merely unfunded in trinity's ledger; on the
published picture it should not exist — and the inconsistency is **internal to trinity**, not just a
difference from the literature. (Pointed out independently in `LITERATURE_ASSESSMENT.md` §4.3.)

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

#### Pre-registered gates for B11.A–D — written 2026-08-18 **before** any of them was run

Registered up front so the numbers cannot be graded after the fact. Each gate names its falsifier.
A run that never reaches the phase a gate needs is **VOID**, never a confirming null.

**B11.A — photon-conserving fixed point.** Formulation: let `x` be the fraction of `Qi` consumed by
the cavity, so the shell receives `(1−x)·Qi`. A cavity Strömgren-filled at
`n(x) = sqrt(3·x·Qi/(4πχ_e α_B R2³))` consumes exactly `x·Qi` **for any `x`** — the cavity balance
alone is one equation in two unknowns and does not close. The shipped code closes it by fiat with
`x = f_abs(Qi)`, i.e. the shell's absorbed fraction computed from the *undepleted* flux. The
photon-conserving closure of the same scheme is the fixed point
> **`x = f_abs(Qi·(1−x))`**, with `f_abs(Q)` = `shell_structure_pure` re-run with `params['Qi'] = Q`.
- **G11.A1 — root structure.** Bisect `g(x) = f_abs(Qi·(1−x)) − x` on `x ∈ [0,1]` per driving row;
  report every root and the shape of `g`. *Prediction:* on rows where `f_abs(Qi)` is already 1.0000
  (29 of 33), `f_abs` is flat in `Q` near the top, so the unique root is `x = 1` — the cavity takes
  every photon and the shell is left **neutral**, which is the degenerate outcome §Batch 11
  pre-registered as "itself the answer". *Falsifier:* an interior root `x* < 0.999` on any driving
  row means the fixed point is non-trivial and the degeneracy reading is wrong.
- **G11.A2 — the ratio the PLAN asked for.** Report
  `P_C3a_fixedpoint / P_C3a_shipped = sqrt(x* / f_abs(Qi))` per phase. *Prediction:* **≥ 1 on every
  driving row**, because `f_abs` is non-increasing in `Q` so `x* ≥ f_abs(Qi)`. ⚠️ **This gate can
  embarrass §6b's seam A and is registered so that it can.** Seam A's *existence* is settled (B11.0
  CONFIRMED the double-spend), but its stated *consequence* — "a photon-conserving accounting has
  less than `Qi` available to the cavity ⇒ `P_C3a` overstated" — predicts a ratio **< 1**. If the
  measured ratio is ≥ 1 throughout, that clause is **wrong** and must be struck from §6b and from
  the "upper bound" list, exactly as seam B's direction was.
- **G11.A3 — confined-branch null.** Energy/implicit rows must give `P_HII` = 0 under both closures.
  Any non-zero fails and invalidates the whole of B11.A.

**B11.B — the boundary/drive inconsistency (re-scoped, see above).** Replay `shell_structure_pure`
on each driving row twice: once at the shipped `params['Pb']`, once with the inner pressure set to
that row's `P_C3a`.
- **G11.B1.** `Δf_abs` = 0 exactly on the 29/33 driving rows where `f_abs` is already 1.0000.
  *Falsifier of B11.0's revision:* any non-zero `Δf_abs` on those rows.
- **G11.B2.** `ΔP_C3a ≥ 0` on **every** driving row. *Falsifier of B11.0's revision:* any row with
  `ΔP_C3a < 0`, which would restore seam B to the "upper bound" list.
- **G11.B3.** Report `Δ dR_ion`, `Δ f_ionised_dust` and `Δ shell_n0` as the size of the
  inconsistency. Descriptive, no pass/fail — this is what B is actually about after the re-scope.

**B11.C — mass ledger consequence.**
- **G11.C1 — supply.** Compare the photoevaporative supply off the shell's ionised face,
  `Ṁ_supply = 4πR2²·n_C3a·mu_convert·c_i` with the **isothermal** `c_i = sqrt(k_B·T_ion/mu_ion_shell)`
  (stated here so it is not chosen after seeing the answer), against the `dM_cav/dt` the shipped
  trajectory demands (central difference of the measured `M_cav(t)`). Report
  `Ṁ_supply / Ṁ_required` per driving row. **Supply is adequate** if the ratio ≥ 1 on ≥95% of
  driving rows; **supply-limited** otherwise. ⚠️ Adequate *rate* is not sufficiency: B11.0 showed
  `shell_mass` already equals 100% of the run's gas, so any real supply must debit `shell_mass`.
  Both must be reported together.
- **G11.C2 — dynamics.** Re-integrate the momentum-phase equation of motion offline from the first
  momentum row with the shell mass debited, `M_eff(t) = shell_mass(t) − M_cav(t)`.
  - **G11.C2a — validity control (blocking).** The *same* integrator run with `M_eff = shell_mass`
    must reproduce the run's own `R2(t=1.5)` to **≤2%**. If it does not, the offline EOM is not
    faithful and **G11.C2b is VOID** — not a null, not a small effect. Reported either way.
  - **G11.C2b.** `ΔR2` at matched `t` = 1.5, debited vs control. *Prediction:* debiting up to 56% of
    the inertia makes the shell faster, so `ΔR2 > 0`, order 10–30%. No pass/fail — this is a
    magnitude measurement of how much the double-book matters, which is what "measure, don't guess
    whether 56% mass matters" asks for.

**B11.D.** Documentation only; the numbers are already in B11.0 (`dR_full/R2` 0.6723–1.3078). The
deliverable is a stated validity limit, not a gate.

#### B11.A–D RESULT — 2026-08-18, measured against the gates above

All three measurement batches ran on the same B3M reproduction. **Two pre-registered gates came
back against what §6b said, and one came back against my own prediction; all three are recorded as
misses rather than reinterpreted.**

| gate | result | consequence |
|---|---|---|
| G11.A1 root structure | **33/33 driving rows resolve to `x* = 1`, zero interior roots** | the photon-conserving fixed point is **degenerate**, as pre-registered |
| G11.A2 `P_C3a` ratio | **1.0000–1.1778, 0/33 rows below 1** | ⛔ §6b seam A's *consequence* clause **REFUTED** — it predicted < 1 |
| G11.A3 confined null | 0 violations / 83 confined rows | PASS |
| G11.B1 `Δf_abs` | 0 non-zero on all 29 saturated rows | B11.0's revision of seam B **holds** |
| G11.B2 `ΔP_C3a` | **+0.0000…+0.1778, 0/33 negative** | B11.0's revision of seam B **holds** |
| G11.C1 supply | ratio **1.32–2.13**, frac ≥ 1 on **100%** of rows | **supply adequate by rate** — the "supply-limited" escape is closed |
| G11.C2a control | **0.871%** against a 2% blocking bar | PASS — G11.C2b is a measurement, not VOID |
| G11.C2b dynamics | **+8.55%** (inertia) / **+9.22%** (full) in `R2` at t=1.5 | sign as predicted, magnitude **below** my registered 10–30% |

**B11.A — the fixed point is degenerate, and fixing the double-spend RAISES `P_C3a`.**
A cavity Strömgren-filled at `n(x) = sqrt(3·x·Qi/(4πχ_e α_B R2³))` consumes exactly `x·Qi` for *any*
`x`: the cavity balance is one equation in two unknowns. The shipped code closes it with
`x = f_abs(Qi)`; the photon-conserving closure is `x = f_abs(Qi·(1−x))`. Gridding
`g(x) = f_abs(Qi(1−x)) − x` and bisecting any bracketed root gives **the endpoint `x* = 1` on every
one of the 33 driving rows, with no interior root anywhere**. So the photon-conserving version of
C3a's own scheme says **the cavity absorbs 100% of `Qi` and the shell is left neutral** — which
contradicts trinity's own shell solve (99.5% ionised in momentum) *and* the ionised-shell boundary
condition that sets `nShell0`. That is the outcome the PLAN pre-registered as "itself the answer":
the scheme cannot be made photon-conserving without a second equation, which is exactly the equation
Geen et al. supply (B11.G).
⛔ **And G11.A2 refutes §6b's seam-A consequence.** `P_C3a_fixedpoint/P_C3a_shipped =
sqrt(x*/f_abs(Qi))` = **1.0000–1.1778 with 0 of 33 rows below 1**. §6b wrote "a photon-conserving
cavity+shell accounting has less than `Qi` available to the cavity ⇒ `P_C3a` overstated", i.e. a
ratio below 1. The opposite is measured: conserving photons *raises* `P_C3a` by up to 17.8%, because
the cavity ends up with **more** than `Qi·f_abs`, not less. **The double-spend itself is untouched**
— B11.0 CONFIRMED it and it remains a genuine defect — but the clause "⇒ `P_C3a` overstated" is
struck, and seam A comes off the "upper bound" list along with seam B.

**B11.B — B11.0's revision survives both of its own falsifiers, and the inconsistency is large.**
Replaying each driving row at the drive's claimed inner pressure instead of `params['Pb']`:
`Δf_abs` = 0 on **all 29** saturated rows (G11.B1) and `ΔP_C3a/P_C3a` = **+0.0000…+0.1778 with zero
negative rows** (G11.B2). Both falsifiers registered against B11.0 failed to fire, so seam B's
direction is settled: **up or nothing, never down.** Size of the inconsistency (G11.B3, descriptive):
`shell_n0` rises **4.70×** in transition and **6.17×** in momentum (it is linear in the boundary
pressure, so this is just `P_HII/Pb` again), the ionised layer **thins by 79–83%**, and the
dust-absorbed fraction of ionising photons falls **0.620 → 0.455** (transition) and
**0.607 → 0.395** (momentum). That last one is the interesting number: the shell's dust sink — G9.4's
−51–75% of the budget — is itself materially different under the two pressures, so seam B and the
dust finding are not independent.

**B11.C — the rate is fine; the reservoir is not; and it is worth ~9%, not ~2×.**
*Supply (G11.C1).* Photoevaporation off the shell's ionised face delivers **5.10e4–8.68e4 Msun/Myr**
against a required `dM_cav/dt` of **2.40e4–6.31e4**, i.e. **supply/required = 1.32–2.13 on 100% of
rows** (isothermal `c_i` = 11.6445 pc/Myr = 11.4 km/s, fixed in the pre-registration). **The cavity
is not rate-limited.** This *closes* the "supply-limited" limb of §6b's either/or rather than
supporting it — but it does not rescue the premise, because B11.0 showed `shell_mass` already equals
100% of the gas the run has. A real photoevaporative flow at this rate is precisely mass moving
**out of the shell**, which is the double-book, stated as a flux instead of a total.
*Dynamics (G11.C2).* The blocking control passed at **0.871%** (offline control 23.0503 pc vs the
run's 23.2527 pc at t=1.5), so the debited numbers are a measurement. Debiting the shell by `M_cav(t)`
gives `R2(t=1.5)` = **25.0218 pc (+8.55%)** debiting inertia only, or **25.1764 pc (+9.22%)** debiting
gravity as well; the two variants bracket how the cavity gas is treated gravitationally.
⚠️ **My pre-registered prediction of "order 10–30%" missed.** The sign held, the magnitude did not —
recorded as a miss, not rounded into range. The reason is legible: inertia enters as `1/m`, and
`M_cav/M_shell` is only 0.10 at the start of the momentum phase, reaching 0.56 only at the very end,
so the trajectory spends most of its length barely debited. **So "does 56% mass matter?" has a
number: ~9% in `R2`, comparable to the 4.0% C1 cost and well below the 12.8–20.5% C3c itself moved.**

**B11.D — stated validity limit (no gate, no fix this batch).**
> The momentum-phase ODE assumes a thin shell, and C3a assumes a sharp cavity/shell split at `R2`.
> On B3M's momentum rows the shell is **not thin**: `dR_full/R2` = 0.6723–1.3078 and
> `dR_ion/R2` = 0.6579–1.3076, i.e. the shell is between two-thirds of and larger than the cavity it
> surrounds, and it is **99.54% ionised**, so "cavity" and "shell" are not distinguishable by
> ionisation state there. Both premises are outside their validity range in the momentum phase, and
> quantitatively so from `dR/R2 ≳ 1/3` onward — which the trajectory crosses *inside the transition
> phase*. This is a stated limit of the current model, not a defect with a proposed fix.

**Where §6b stands after B11.0 + B11.A–D.** Of the four seams, **all four exist**; but the
"every seam pushes the same way, `P_C3a` is an upper bound" summary has now lost **two** of its
three members. What remains on the upper-bound list is **seam C** (the filled-cavity limb needs 56%
more gas than exists, so the drive must be the supply-limited one) and **G9.4's dust** (−51–75% of
the photon budget). Seams A and B are real inconsistencies whose repair, measured, moves `P_C3a`
**up** by 0–18%. The honest one-line summary is therefore: *the driving-branch `P_C3a` is bounded
above by the mass ledger and the dust sink, not by the photon or boundary bookkeeping* — and the
photon and boundary bookkeeping cannot be repaired within C3a's structure at all, because its
photon-conserving fixed point is degenerate (G11.A1). That is the strongest argument yet that D5
needs a different closure rather than a patched C3a, and B11.G names a published one.
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
CONFIRMED, B REVISED, none REFUTED) ~~AND A–C are quantified~~ (**DONE 2026-08-18**, see the B11.A–D
RESULT above). **Both conditions are now met, so the hold's stated release criteria are satisfied —
but the release is the maintainer's call, not this document's**, and the numbers argue for a
different decision than the one D5 was framed around: seams A and B cannot be repaired inside C3a
(its photon-conserving fixed point is degenerate), so D5's question is no longer "C3c-switch vs
C3a-raw" but "C3a at all". **B11.G is the recommended input to that call** and is cheap. B11.F is
housekeeping.

### Batch 12 — the low-wind rung: old-vs-new where P_HII must dominate — Status: ✅ **DONE 2026-08-18 — G12.1/G12.2/G12.4 PASS; G12.3 fired my own seam-C falsifier**

**Maintainer question (2026-08-18):** *"does the current data include low wind regimes where P_HII
are sure to dominate? this helps compare the most current improved implementation against the old
version, as a sanity check."*

**Answer: no — not in a form that can carry the check.** Inventory of every committed CSV, by config:

| what exists at low wind | file | why it cannot serve |
|---|---|---|
| `B3MW01` (`Lw × 0.1`), 4 phases, momentum 100% HII-dominated | `data/b5s3_ladder_regime.csv`, `_screen`, `_lag` | built by `c3_offline_screen.py` — an **offline screen evaluated on STOCK trajectories**. It predicts what C3c *would* give on a stock `R2(t)`; it is not a C3c-arm run, so it cannot be the "new implementation measured" |
| `B3MW001` (`Lw × 0.01`), 78.4% HII-dominated in **energy** | `data/b7_confinement_screen.csv` | `run_complete = False`, `PARTIAL_in_progress` — reaches energy+implicit only and **never enters transition or momentum**. VOID for any driving-branch claim, per this plan's own rule |
| `WW` (`FB_thermCoeffWind = 0.1` on `simple_cluster`) | `data/b6_ledger.csv`, `b0_trajectories.csv`, `b5_c3c_regime.csv` | present on both arms, but `b6_ledger` is a matched-`t` ΔR2/fate row, not per-row `P_HII`; and `b0`'s arm is stock-only |

And the two structural gaps behind it:
- **The only paired stock-vs-C3c full trajectory in the workstream is `data/b7_regime_trajectory.csv`,
  and it is `B3M` only** (`Lw = 1`).
- **The entire replay family runs strong-wind-only.** `data/b10_wind_profile.csv` covers
  `B3M / B3MW3 / B3MW10` — i.e. `Lw` ∈ {1, 3, 10}. Every Batch 11 diagnostic
  (`b11_mass_ledger`, `b11_photon_ledger`, `b11_mass_dynamics`) is **`B3M` alone**. The wind ladder
  in the measured data only ever goes *up*.

**Why this is the right sanity check, and not just another rung.** Four independent reasons:
1. **It is where old and new differ maximally, by construction.** The old `P_HII` was the capped
   Strömgren density, which Batches 0/1 measured to be an exact algebraic relabelling of the
   confining pressure — identity to ≤2.9e-16 with the cap binding on **100% of rows in every phase**
   of 6 configs across 4 decades of `nCore`. So the old code has `P_HII/Pb ≡ 1` and **can never show
   photoionisation dominating at any wind strength**. The new one carries `Qi` and `R2` and should
   dominate hardest exactly when the wind is weakest. A weak-wind rung is therefore the cleanest
   possible discriminator between the two implementations.
2. **The literature says the answer in advance.** Lancaster Paper I `eq:Rch_def` gives
   `R_ch ∝ ṗ² α_p² / Q_0`, so weak wind ⇒ **small `R_ch`** ⇒ `R_w/R_ch ≫ 1` ⇒ PIR-dominated, and
   their coupled force `F_b = α_p ṗ (1 + R_w/R_ch)^{2/3}` → `F_b,Sp`. Geen et al. reach the same
   ordering via `C_w ∝ ṗ_w^{3/2} Q_H^{-3/4}`. So this is a regime with a **published expected
   answer**, not just an untested corner.
3. **It is the regime where C3a's unconfined branch should be asymptotically RIGHT.** Per
   `LITERATURE_ASSESSMENT.md` §4.2 (verified there numerically), the CEM force tends to the pure
   photoionised limit as `R_w/R_ch → ∞` to 0.05% — which is C3a's unconfined branch. Batch 11 found
   the seams at `Lw = 1`, near the crossover; low wind tests whether they persist where the branch is
   supposed to be exact.
4. **It isolates C3a from `alpha_p`.** `LITERATURE_ASSESSMENT.md` §4.1 hypothesises that the
   universally HII-dominated momentum phase is really a missing `α_p ≈ 5–6` on `P_ram`. At `Lw × 0.1`
   the wind term is small *whatever* `α_p` is, so a low-wind rung cannot be explained away by it.
   Any HII dominance measured there is a statement about C3a alone.

**Runs launched 2026-08-18 (both arms, `B3MW01` = `Lw × 0.1`, `stop_t` 1.5):**
```
# new (C3c, this branch)
python docs/dev/phii-identity/harness/run_batch.py --arm b11lowwind --configs B3MW01 \
    --stop-t 1.5 --root <scratch>/runs/b11lw
# old (pre-C3c): git worktree at fca7d88e, with the CURRENT harness copied in
#   (fca7d88e's run_batch.py predates --root; the harness is docs/, not trinity/,
#    so copying it does not change the physics under test)
python docs/dev/phii-identity/harness/run_batch.py --arm b11lowwind_stock --configs B3MW01 \
    --stop-t 1.5 --root <scratch>/runs/b11lw_stock
```

**Pre-registered gates — written before either run finished.** ⚠️ A run that does not reach the
phase a gate needs is **VOID** for that gate, never a confirming null. `B3MW001` is the cautionary
precedent: it looks like low-wind coverage and is not.

- **G12.1 — the old identity, re-confirmed off its measured grid.** Stock arm: `P_HII/Pb` = 1 to
  ≤1e-12 on ≥99% of rows in every phase. Batch 0 established this on 6 configs but **never at
  `Lw × 0.1` on this cloud**, so it is a genuine out-of-sample test of the identity claim, not a
  restatement. C3c arm: `P_HII` exactly 0.0 on confined rows, `> Pb` on driving rows.
  *Falsifier:* any stock row off unity beyond the known stale-`Pb` handoff rows.
- **G12.2 — does P_HII actually dominate at low wind, in a real run?** C3c arm, `frac_HII_dominated`
  per phase. The offline screen (`data/b5s3_ladder_regime.csv`) predicts **transition 0.9118,
  momentum 1.0000, `drive_ratio` 6.68–7.31**. That screen was evaluated on the *stock* trajectory,
  so a C3c arm — which moves `R2` — can disagree; this is a real test of the screen as a predictor at
  low wind, where it was only ever validated at nominal wind (Batch 5 stage 2).
  *Pass:* measured momentum `frac ≥ 0.9`. *Informative failure:* a large miss retires the offline
  screen as a low-wind predictor.
- **G12.3 — are the Batch 11 seams a strong-wind artefact?** Re-run all three B11 harnesses on the
  low-wind C3c arm and report seam C (`M_cav/M_shell`, `(M_cav+M_shell)/M_avail`), seam A (the
  fixed-point ratio and root structure) and seam D (`dR/R2`) at matched `t`.
  *Prediction:* **seam C is at least as bad at low wind.** `M_cav ∝ R2^{3/2}·sqrt(Qi f_abs)` and `Qi`
  is untouched by the wind knob, while a weaker wind hands more of the drive to the HII term.
  *Falsifier of "the seams generalise":* seam C's `M_cav/M_shell` at t = 1.5 falling below 0.2
  (vs 0.5638 at nominal wind) would make the mass double-book regime-specific and materially weaken
  §6b seam C.
- **G12.4 — trajectory cost, old vs new.** ΔR2 at matched `t`, stock vs C3c, against the nominal-wind
  range the b6 ledger recorded (7.6–20.5% over 13 configs). *Prediction:* **larger** than nominal,
  since `P_HII` is a bigger share of the drive when the wind is weak. Magnitude only, no pass/fail.

**What Batch 12 does NOT do.** It changes no `trinity/` source, and it does not test `alpha_p` — that
is `LITERATURE_ASSESSMENT.md` §4.1's offline screen, which is a separate, cheap piece of work and
should be run on these same trajectories once they exist.

#### Batch 12 RESULT — 2026-08-18, measured against the gates above

Both arms completed: C3c **702 s / 205 snapshots**, stock **666 s / 222 snapshots**, both reaching
`t = 1.5` in the **momentum** phase, so **nothing here is VOID**. Fate unchanged on both
(`stopping_time`). Phase row counts move as expected for a stronger drive — energy 69/69,
implicit 79/78, transition 32/34, momentum 25/41 (c3c/stock).

| gate | result | verdict |
|---|---|---|
| G12.1 old identity, stock arm | `\|P_HII/Pb − 1\|` ≤ **4.44e-16**; frac within 1e-12 = **1.0000** in implicit/transition/momentum | **PASS** |
| G12.1 new branch behaviour | `P_HII` = 0.0 on **69/69** energy and **79/79** implicit; driving rows `P_HII/Pb` = 1.240–14.369, **0 rows ≤ 1** | **PASS** |
| G12.2 does P_HII dominate? | transition **0.9062** (screen: 0.9118), momentum **1.0000** (screen: 1.0000) | **PASS** (bar ≥ 0.9) |
| G12.3 seam A | 27/27 endpoint root `x* = 1`; fixed-point ratio **1.0000–1.0000**; null 0/76 | generalises |
| G12.3 seam B | Δ`f_abs` = 0 on 27/27; Δ`P_C3a` ∈ [−1.46e-16, +2.37e-16], **0 rows** past the 1e-14 roundoff floor | revision **holds**; inconsistency **bigger** |
| G12.3 seam C | `M_cav/M_shell` = **0.1296** at t=1.5 | ⛔ **my registered falsifier FIRED** |
| G12.3 seam D | momentum `dR_ion/R2` = **1.171–1.438** (median 1.213) | **worse** than nominal |
| G12.4 trajectory cost | ΔR2 = **+35.138%** at matched `t`, fate unchanged | prediction **held** |

**The sanity check itself — this is the cleanest old-vs-new demonstration the workstream has.**
In the one regime where photoionisation *must* dominate, the two implementations say opposite things
about the same cloud:

| | old (pre-C3c, `fca7d88e`) | new (C3c, this branch) |
|---|---|---|
| momentum `P_HII/Pb` | **1.0** to 2.2e-16 | **13.667–14.369** |
| what `P_HII` depends on | nothing — it is `Pb` returned | `Qi`, `f_abs`, `R2` |
| `frac_HII_dominated`, momentum | **0** (it cannot exceed `Pb`; they are equal) | **1.0000** |
| `R2(t=1.5)` | 5.722 pc | **7.733 pc** |

The old code returns the confining pressure to **2.2e-16** even when the wind has been cut by 10×,
i.e. it carries **no photoionisation information at any wind strength** — which is the defect this
workstream exists for, now shown in the regime where it matters most rather than only at nominal
wind. **G12.1 is also the identity's first out-of-sample confirmation**: Batch 0 measured it on six
configs but never at `Lw × 0.1` on this cloud, and it reproduces to the same 2e-16.
(The one exclusion is the **energy** phase, 0.9855 within 1e-12, max deviation 6.33e-02 — the
documented stale-`Pb` rows at the 1a→1b handoff that Batch 1 already corrected, not cap-slack.)

**The offline screen is validated at low wind.** `data/b5s3_ladder_regime.csv` predicted transition
0.9118 / momentum 1.0000 from the *stock* trajectory; the real C3c arm gives **0.9062 / 1.0000**,
i.e. 0.6% relative in transition and exact in momentum — despite the arm's `R2` being 35% larger.
The screen was only ever validated at nominal wind (Batch 5 stage 2); it now holds a decade below.
Momentum `P_HII/Pb` = 13.667–14.369 against nominal wind's 5.091–7.156 (median 6.165) is a factor
**2.2×**, consistent with stage 3's `Lw^−0.33` (which predicts 2.14×; the measured exponent is 0.356).

⛔ **G12.3 seam C — my pre-registered falsifier fired, and §6b's headline number is now regime-scoped.**
I registered: *"seam C's `M_cav/M_shell` at t = 1.5 falling below 0.2 (vs 0.5638 at nominal wind)
would make the mass double-book regime-specific and materially weaken §6b seam C."* Measured:
**0.1296**. It fired, and I am recording it as fired rather than reinterpreting the bar.
**What is wrong is my reasoning, not the bar.** I predicted the seam would worsen because a weaker
wind hands more of the drive to the HII term. But `M_cav ∝ R2^{3/2}·sqrt(Qi f_abs)`, so the
controlling variable is **bubble size, not the degree of HII dominance** — and the low-wind run only
reaches `R2 = 7.733 pc` against 23.253 pc. The ratio of `R2^{3/2}` is 0.192 and the measured mass
ratio is 12,966/57,397 = **0.226**; the rest is `f_abs`. So:
> **The mass double-book is worst in the configs that expand furthest, not in the configs where
> photoionisation dominates most.** §6b seam C's `M_cav/M_shell → 0.564` is a **B3M number**, not a
> universal one, and must be quoted with the config.

**What survives the falsifier, and it is the load-bearing half.** The *qualitative* result is
untouched at low wind: `(M_cav + M_shell)/M_avail` = **1.1296**, so the model still asserts 13% more
gas than the run has; `M_cav/M_bubble` = **410**, and the wind injects **31.6 Msun** = 0.24% of
`M_cav`. There is still no supply — there is just less of it to not-supply. The dynamical cost falls
with the mass: **+0.446%** (inertia) / **+0.965%** (full debit), against a control passing at
**0.213%**. So at `Lw × 0.1` the double-book is worth ~1% in `R2`, not ~9%.

**Seams A, B and D do NOT follow C.** This is the useful structural result of Batch 12:
- **A generalises exactly.** 27/27 driving rows give the endpoint root `x* = 1` — the degeneracy is
  not a nominal-wind artefact — and the fixed-point ratio is **1.0000–1.0000**, because `f_abs` is
  saturated at 1 on *every* driving row here. Seam A's repair still never lowers `P_C3a`.
- **B holds and gets bigger.** Both B11.0 falsifiers stayed clear (Δ`f_abs` identically 0 on 27/27;
  the single negative Δ`P_C3a` is **−1.46e-16 = 0.66 ULP** on a row with Δ`f_abs` exactly 0, i.e.
  roundoff in the `sqrt`, not a sign reversal — the harness now separates the two rather than
  printing it as a trip). The inconsistency itself **doubles**: `shell_n0` ratio **11.84**
  (transition) / **13.77** (momentum) against 4.70/6.17 at nominal wind, the ionised layer thins
  **88–90%** against 79–83%, and the dust-absorbed fraction moves 0.663→0.467 and 0.658→0.459.
- **D is worse.** Momentum `dR_ion/R2` = **1.171–1.438** (median 1.213) against 0.658–1.308
  (median 0.975). The thin-shell premise is *more* violated at low wind, not less.

**Net for §6b.** Three of the four seams are confirmed regime-robust or regime-worsening; **only
seam C is regime-scoped**, and its qualitative core (no supply, over-subscription) survives anyway.
Since seam C is the *only* member left on the "upper bound" list after B11.A–D, the practical
consequence is that **the size of the upper-bound correction is config-dependent and largest in the
widest-expanding runs** — which is exactly where the momentum question lives, so the conclusion is
unchanged for the case that motivated it.


### Batch 13 — K10 offline screen: the smooth CEM drive on committed trajectories — Status: ✅ **DONE 2026-08-27 — G13.2 PASS; G13.1/G13.3 FAILED by my own design error (diagnosed); G13.4's dust rule fired**

**Where K10 comes from.** The maintainer asked whether the confined branch's exact 0.0 could be
replaced by "an approximation much closer to truth" that also removes the sudden `P_HII` influx at
the switch. The literature supplies exactly that (Lancaster's coupled closure; independently the
Geen 2022 structure), and B11.G rung 0 already verified its identities. K10 (§7.1) is its trinity
form. This batch measures it **offline only** — no `trinity/` change, ship-hold intact.

**Inputs (all committed):** `data/b7_regime_trajectory.csv` (B3M, c3c arm) and
`data/b12_lowwind_trajectory.csv` (B3MW01, c3c arm) — both carry `t_now, current_phase, R2, Qi,
shell_fAbsorbedIon, Pb, P_ram, P_HII`. Dust variant joins `data/b11_photon_ledger.csv` /
`data/b12_lowwind_photon_ledger.csv` (`dust_Pb` per driving row) by nearest `t`.

**Pre-registered mapping** (fixed before running): `P_conf` = `P_ram` (momentum), `max(Pb, P_ram)`
(transition), `Pb` (energy/implicit — the stored, un-ramped value; rows with `t ≤ 1e-3` are
excluded as the `dt_switchon` window and counted). Shipped comparator from stored columns:
`P_HII + P_ram` (momentum), `max(Pb, P_HII + P_ram)` (transition), `max(Pb, P_HII)` (energy/
implicit). Rows missing a needed column are skipped and counted; a config contributing <5 driving
rows is **VOID** for gates, never a confirming null.

**Pre-registered expectation (disclosed, not a gate):** back-of-envelope in the design discussion
put the no-dust K10 momentum drive at `(1+x)^{2/3}` ≈ 9.5–14.7 `P_ram` on B3M — **above** the
shipped sum's 6.1–8.2 — with the dust-corrected variant landing back near it. The screen exists to
measure this properly; recording the expectation up front keeps the measurement honest.

**Gates:**
- **G13.1 — continuity (the point of K10).** At every row where the shipped `P_HII` crosses 0 →
  positive (the C3c switch), the K10 drive's step between adjacent rows is **< 5%**, against the
  shipped jump (measured 23.4% in transition; exact factor 2 at a momentum-phase switch).
  *Falsifier:* any K10 step ≥ 5% at a switch row.
- **G13.2 — the healthy branch stays healthy.** On B3M energy+implicit rows the K10 drive's median
  excess over the shipped drive is **≤ 15%**. *Falsifier:* median > 15% — K10 would then be
  re-deciding a branch §6b found exactly consistent, an informative failure.
- **G13.3 — implementation self-consistency (MD identity).** On momentum rows the `P_conf`-form
  equals the `(1 + R2/R_ch)^{2/3}`-form to **1e-10** relative. *Falsifier:* any row worse.
- **G13.4 — dust sensitivity.** Recompute with `Q_eff = Qi_abs·(1 − f_dust,ion)` on rows where the
  ledger provides dust. Pre-registered rule: if the two variants differ by **> 2×** in the momentum
  phase, the verdict is **"K10 cannot ship without a dust model"** — that outcome is itself the
  answer, not a failure of the screen.
- **G13.5 — magnitude (measurement, no bar).** K10/shipped drive per phase per config, both
  variants, plus predicted `R_i/R2` vs the measured `R_IF/R2` where layer data exists.

#### Batch 13 RESULT — 2026-08-27, measured against the gates above

304 rows screened (B3M 156, B3MW01 148; 132 skipped as `dt_switchon` window / missing column /
`P_conf ≤ 0`). **Two of the five gates FAILED, both by MY OWN design error, and both are recorded
as failed with the diagnosis rather than re-barred.**

| gate | result |
|---|---|
| G13.1 continuity | ⛔ **FAIL as written** (worst K10 row-step 7.97% vs a 5% bar) — **metric was wrong**, see below |
| G13.2 healthy branch | ✅ **PASS** — B3M energy+implicit median K10 excess **+0.68%** (bar 15%) |
| G13.3 MD identity | ⛔ **FAIL** (6.53e-2 vs 1e-10) — **a real convention discrepancy**, diagnosed exactly |
| G13.4 dust sensitivity | ⛔ fired: momentum A/B = **2.05×** > 2× ⇒ **"K10 cannot ship without a dust model"** |
| G13.5 magnitude | measured, below |

**G13.1 — the gate was mis-specified, and the property it meant to test holds exactly.** I
registered "the K10 drive's *step between adjacent rows* is < 5% at a switch row". Adjacent
snapshots are separated by real evolution, so that metric conflates the discontinuity with genuine
change in `R2`, `Qi` and `Pb` — the *shipped* rule scores only 5.97–6.79% by the same metric, which
is the tell that the metric is not measuring a jump. Re-measured correctly, as a jump **at fixed
state** (evaluate the shipped rule with `P_HII = 0` and with `P_HII = P_C3a` at the same row):

| config | t | phase | below | above | shipped JUMP | K10 jump |
|---|---|---|---|---|---|---|
| B3M | 0.3012 | transition | 1.3494e3 | 1.8079e3 | **+34.0%** | **0.0%** |
| B3MW01 | 0.7186 | transition | 1.2907e3 | 1.7189e3 | **+33.2%** | **0.0%** |

K10 is a single-valued smooth function of `(R2, Qi_eff, P_conf)` with no branch, so its state-jump
is **identically zero** by construction. The claim K10 was registered to support is confirmed; the
gate that was supposed to test it was badly designed. (Note the shipped state-jump measures **+34%** at the first
post-crossing snapshot. ⛔ **This sentence's dismissal of the 23.4% `P_ram/Pb` figure is WITHDRAWN
2026-08-29 (Batch 20 slice 0):** 23.4% is the scheme's *intrinsic* discontinuity at the crossing
(`max(Pb, P_ram)` → `max(Pb, Pb + P_ram)` = `1 + P_ram/Pb`, analytic), while +34% is that jump plus
one sampling interval of evolution — the same conflation that got G13.1's original metric thrown out.
Both are real; they answer different questions.)

**G13.3 — a genuine convention discrepancy, and it propagates backwards.** The two K10 forms
disagree by exactly `chi_e^{2/3}`: my primary form carries trinity's explicit electron factor
(`chi_e * alpha_B * n_H^2`, as in `shell_structure.py:247` and `get_phii_c3c`), while Lancaster's
`eq:ionreceq2` writes `alpha_B n_H^2` with **no** separate electron factor. Predicted asymptotic
relerr `1.1^{2/3} − 1 = 0.0656`; measured **0.0653**. So:
> **`R_ch`(trinity convention) = `chi_e` × `R_ch`(Lancaster).**

The primary form is the correct one *for trinity*, because it matches the code's own recombination
convention. ⚠️ **This invalidates a number recorded earlier the same day**: the old-vs-new-vs-CEM
comparison in the §9 verdict entry used Lancaster's `R_ch` without `chi_e`, so its `F_CEM` is high
by `1.1^{2/3}` = 6.6%; `new/CEM` = 0.548–0.638 becomes **0.583–0.679**, `old/CEM` = 0.134–0.210
becomes **0.143–0.223**. **No conclusion changes** — old remains 3–7× low under every mapping — but
the figures are corrected here. (`harness/cem_closure_check.py` is unaffected: it is explicitly
scale-free with `chi_e` folded into `alpha_B`.)

**G13.4 — the pre-registered dust rule fired.** Momentum-phase sensitivity between
`Q_eff = Qi·f_abs` (A) and `Qi·f_abs·(1 − f_dust,ion)` (B) is **2.05×** on B3MW01 and 1.89× on B3M,
above the 2× bar. **Verdict as registered: K10 cannot ship without a dust model.** This is the
outcome the rule anticipated, not a failure of the screen — and it is consistent with G9.4, which
measured the dust sink at 61–75% of the absorbed budget.

**G13.5 — magnitudes, and the disclosed expectation held.** Momentum drive in units of `P_ram`:

| config | shipped | K10 no-dust (A) | K10 dust-corrected (B) |
|---|---|---|---|
| B3M | 6.1–8.2 | **9.0–14.0** | **4.9–8.5** |
| B3MW01 | 14.7–15.4 | **32.8–35.0** | **16.0–17.9** |

The pre-registered expectation ("no-dust K10 ≈ 9.5–14.7 `P_ram` on B3M, above the shipped 6.1–8.2")
is **confirmed** — 9.0–14.0 measured. Median ratios K10/shipped: B3M momentum 1.605 (A) / **0.851**
(B); B3MW01 momentum 2.242 (A) / **1.096** (B); transition 1.381/0.794 and 2.074/1.059.
Predicted `R_i/R2`: 3.39 (A) → 2.39 (B) on B3M momentum, against the shell solve's measured
`R_IF/R2` ≈ 1.7–2.3 — **the dust-corrected variant lands on the measured layer geometry**, the
no-dust one does not.

**The substantive finding.** The **dust-corrected** coupled closure sits within **15%** of the
shipped C3c drive in B3M's momentum phase (0.851) and within **10%** on B3MW01 (1.096), while the
no-dust form is 1.6–2.2× above it.

⛔ **The "cancellation" explanation offered here is WRONG and is retracted (2026-08-29, maintainer
asked it be checked; `harness/cancellation_check.py`, `data/b19_cancellation.csv`).** It read: *"C3a's
cavity-volume error (which inflates, per K5) and its missing dust sink (which deflates) are of
similar size and opposite sign in this regime."* Both clauses fail, and in different phases:
- **In momentum — the regime the claim named — the two corrections push the SAME way on 17/17
  rows.** `f_volume` = **0.3862**, `f_dust` = **0.6269**; they **compound** to ×0.2506. They cannot
  cancel. Transition likewise compounds on 14/21 rows (0.6475 × 0.6202 = 0.4016).
- **Where the signs ARE opposite — energy and implicit — the magnitudes are nothing like
  "similar":** `f_volume` = **31.98** / **11.49** against `f_dust` = 0.497 / 0.907, netting ×15.4
  and ×10.3. That is a 10–15× inflation, not a cancellation.
- **Mechanism, which the claim missed:** the volume correction changes sign at
  `R_IF/R2 = 2^{1/3} ≈ 1.26`, where the cavity-excluded layer volume equals the cavity volume.
  Measured `R_IF/R2` = 1.000 (energy), 1.003 (implicit), 1.502 (transition), 1.975 (momentum) — so
  the thin-layer phases sit below break-even and invert. This is the same thickness-dependent sign
  Batch 10 found for the geometry correction, and it was already on record.
- **And the explanation does not even connect to what it was explaining.** C3a corrected for BOTH
  errors gives momentum `P_HII/Pb` = **1.545**, against K10's **6.333** — a factor **4.10** apart.
  K10's proximity to the shipped drive is therefore *not* a story about C3a's errors cancelling;
  **K10 is a structurally different closure** whose density comes from pressure equilibrium and
  whose drive carries the `(R_i/R2)²` area amplification, not a corrected C3a.

🔑 **Identity found while checking (not an independent confirmation — algebra).** Corrected-C3a and
the K5b **profile form** agree to the printed digit (1.545 vs 1.545) because they are the *same
quantity*: `n_rms/n_cav = sqrt[(recomb/Qi_abs)·(V_cav/V_layer)] = f_dust · f_volume` exactly. So
**K5b IS C3a corrected for volume and dust**, which unifies G9.4's profile form with this factorial
decomposition and explains why both land on 1.545.

What survives from the original paragraph is only the empirical part: C3c and dust-corrected K10
land within 10–15% of each other in this regime, and that agreement should not be relied on outside
it — but the reason is unexplained, not "compensating errors".

**What Batch 13 does NOT establish.** It is an offline screen on committed trajectories: every row
uses the *shipped* run's `R2(t)`, so it cannot say what K10 would do to a trajectory it drove
itself. The energy/implicit rows are dominated by the confined branch where K10 and shipped agree
to <1% on B3M (but +22–26% on B3MW01 — the low-wind confined branch is where they diverge most,
and G13.2's bar was registered for B3M only). No fate, no ΔR2, no `trinity/` change.

### Batch 14 — K5, the balance volume — Status: 🟡 **offline screen DONE 2026-08-28 — G14.0 passes by the letter but its own diagnostics show the coupling the gate was hunting; no arm run (SHIP-HOLD); scope decision now with the maintainer**

**The change.** `get_phii_c3c` balances `Qi_abs` over the wind cavity `(4/3)πR2³`. Every other
treatment — Lancaster `eq:ionreceq2`, Geen 2019, Geen 2022, **and `shell_structure.py:243`'s own
`_vol_ion = R_IF**3 - rShell0**3`** — balances over the cavity-**excluded** layer. K5 is the one-line
denominator swap that makes `get_phii_c3c` use the volume trinity already uses two files away.

**Two variants, both to be measured; they are not the same change.**
- **K5a (analytic layer):** `n = sqrt(3·Qi_abs / (4π χ_e α_B (R_IF³ − R2³)))`. The literal swap.
- **K5b (profile):** `n = n_rms` over the ionised layer from the shipped solve. G9.4 showed the
  analytic form overstates the real profile's recombination-equivalent density by up to 3.17×, so
  K5b is the more faithful one — but it is a *different kind* of change (it reads the solved
  profile rather than closing a Strömgren balance).

**Conventions pinned now, because a convention mismatch is exactly how G13.3 failed.**
`χ_e` is carried explicitly (trinity writes `χ_e·α_B·n_H²`; Lancaster's `eq:ionreceq2` does not, and
`R_ch`(trinity) = `χ_e`·`R_ch`(Lancaster)). `R_IF` is `rShell_arr_ion[-1]`. Volumes are **exact**
spherical shells, never the thin-shell `4πR²dR` — B11.0's S1 is precisely that mistake, worth
1.34–1.70× in momentum.

**Gates.**
- **G14.0 — decoupling (BLOCKING, run first).** The original defect was `P_HII ≡ P_conf`. K5b reads a
  profile whose *inner boundary* is `nShell0 ∝ Pb`, so re-introducing the circularity is a live
  risk. Regress `P_HII` on `Pb` over driving rows (`harness/coupling_regression.py` already does
  this, Batch 3b). **FAIL if slope ∈ [0.95, 1.05] with r² > 0.99** — that is the old identity
  returning. Reported for K5a and K5b separately; a failing variant is dropped, not re-barred.
- **G14.1 — use the code's own quantity.** The new denominator must equal `shell_structure.py:243`'s
  `_vol_ion` to **1e-12** relative on every row. This is what makes K5 an internal-consistency fix
  rather than a new model. *Falsifier:* any row worse.
- **G14.2 — limits preserved.** `Qi → 0` still returns exactly 0.0. Batch 8's Spitzer/Hosokawa–
  Inutsuka cross-check (`test_phii_c3c_spitzer.py`) either still passes, **or** its expected shift is
  derived and documented *before* re-baselining — the photo-only limit is the one anchor C3a has.
- **G14.3 — magnitude (measurement, no bar).** `P_HII/P_ram` per phase per config for shipped / K5a /
  K5b. Prior expectation, disclosed: momentum 6.165 → ~2.4 (K5a, exact-volume) → ~1.545 (K5b).
- **G14.4 — full-run equivalence (CLAUDE.md rule 5).** Separate processes, matched `t`, on
  `simple_cluster` + `f1edge_{lowdens,hidens}` + B3M + B3MW01. Fate table before/after. This is an
  ODE right-hand side, so a per-call check is **necessary but not sufficient**.
- **G14.5 — does B11.E fall out?** If K5 makes `get_phii_c3c` and `n_IF_Str` the same balance, the
  vestigial `n_IF_Str > 0` gate becomes meaningful again rather than dead. Report; no bar.

#### Batch 14 OFFLINE SCREEN RESULT — 2026-08-28, measured against the gates above

Reproduce: `python docs/dev/phii-identity/harness/k5_offline_screen.py --out
docs/dev/phii-identity/data/b14_k5_screen.csv`. No solver run: B3M is
`b9_layer_density.csv` × `b11_mass_ledger.csv` joined on `row_idx` (two B3M realisations,
per-row agreement re-verified at ≤1.5e-7 on `t`/`R2`, against B11.0's ≤3.3e-6 bound);
B3MW01 is `b12_lowwind_photon_ledger.csv`'s 27 replayed driving rows, **K5a only — no
committed `n_rms` exists for B3MW01, so the K5b low-wind leg is uncovered**, and the
B3MW01 leg sees no confined rows (the photon ledger never replayed them). K5a uses
**exact** spherical volumes (`n_cavity·sqrt(R2³/(R_IF³−R2³))`), not
`b9_layer_density.csv`'s thin-shell `n_layer_analytic` (B11.0 S1). Route check:
reconstructed `P_C3a` matches stored `P_HII` on all 33 B3M driving rows to ≤3.5e-7.

- **G14.0 — no variant trips the bar as registered** (linear slope ∈ [0.95, 1.05] with
  r² > 0.99, shipped-driving rows): K5a slope 1.853/r² 0.949, K5b 1.267/0.977 (B3M);
  K5a 1.730/0.960 (B3MW01). Recorded as **pass by the letter**. ⚠️ **But the disclosed
  log-log diagnostic shows exactly the coupling the gate was written to catch, in a form
  the linear bar cannot see:** B3M K5a slope **+1.016 (r² 0.988)**, K5b **+1.020
  (r² 0.993)** — the candidate values are a **constant multiple of `Pb`**
  (`P ∝ Pb^1.0`; gain ≈2.4 and ≈1.5), where the shipped C3a shows 0.684 (r² 0.918),
  the independent `R2^{-3/2}`-vs-`R2^{-2}` scaling. The gate's bar was aimed at the
  *exact* identity (slope 1, gain 1); what returns is the **identity with an O(1)
  gain** — same defect class, softened. Not re-barred; flagged for the maintainer.
  Independent corroboration this is not a screen artifact: **K5a is algebraically the
  code's own uncapped `n_IF_Str_raw`** (`shell_structure.py:247-251` computes the same
  balance over the same `_vol_ion`; the definitions differ only in `f_abs` vs
  `(1−f_esc_ion)` bookkeeping), and **Batch 3b already measured that quantity's
  coupling on 788 pre-C3c rows: `slope_nIFraw_vs_n0` = 1.005–1.096 at r =
  0.9971–0.9997** (`data/b3b_coupling_regression.csv`). §3b's original finding — *the
  coupling runs through the ionised volume, because `shell_n0 ∝ Pb` is the shell ODE's
  inner boundary* — applies to K5 verbatim: **the layer volume is the coupled
  quantity; the cavity volume is what buys C3a its independence.**
- **Branch census (measurement, disclosed):** under both variants **83/83 confined
  B3M rows flip to driving** (K5a median 2.248×`Pb`, K5b **1.073×`Pb`** in energy —
  K5b returns the old identity to 7% on the confined branch), first flip at
  t = 8.3e-7, **inside the `dt_switchon` window**. A bare denominator swap therefore
  re-admits the D-ramp defect class C3c closed, in every phase — and its full-run cost
  class is already on record: **Batch 4a** (the uncapped layer-volume density in the
  old composition) measured ΔR2_max 15.3–28.4% with the breaches concentrated in the
  ramp window. 0/33 (B3M) and 0/27 (B3MW01) driving rows flip confined.
- **G14.3 — magnitude (no bar), median `P_X/Pb` per phase** (`Pb` IS `P_ram` in
  momentum): B3M momentum **6.165 → 2.381 (K5a) → 1.545 (K5b)** — both priors from the
  gate text hit to the printed digit; transition 3.952 → 2.559 → 1.587; B3MW01
  momentum 13.766 → 4.388 (K5a). Energy/implicit, currently 0 by the switch, would
  come alive at 2.25/1.07 (B3M medians) if the swap kept C3c's branch logic.
- **G14.1/G14.2/G14.4/G14.5 — deferred to the arm** (need the live helper / runs).
  Note for G14.2: Batch 8 proved the Spitzer/HI anchor **on the cavity volume**, so
  K5a shifts the one exact external anchor C3a has; the expected shift must be derived
  before any re-baseline.

**Consequence for the batch.** The screen answers the batch's own premise: `K1/K2 are
internally inconsistent with n_IF_Str` is true, but the *resolution direction* was
backwards — `n_IF_Str`'s layer balance is the coupled (relabelling) form, and
`get_phii_c3c`'s cavity form is the decoupled one. The literature's layer volume lives
inside **coupled closures** (Geen/Lancaster: `P = P_conf·(R_i/R2)²` — the coupling is
explicit and intended), not inside an independent-pressure slot; K10's Batch 13 screen
is that construction and lands within 10–15% of shipped **with dust**. Three ways to
proceed, the choice is D5-adjacent and the maintainer's: (a) run the registered bare
swap anyway to have the C3c-based full-run number (predict: Batch-4a-class early-window
shifts + weaker driving branch); (b) a driving-branch-scoped K5 (keep 0.0 confined,
swap volume only where the switch already fires — momentum drive ÷2.6/÷4) — a design
change beyond the registered one-line swap, needs its own gate; (c) drop K5 as an
independent-value fix and pursue the volume through K10/K6 where the coupling is
structural. The screen cannot pick; it rules out treating (a) as "the minimal
internal-consistency fix" — it is a physics change with a known defect-class cost.

---

### Batch 15 — K9, the shell-mass debit — Status: 🟡 gates registered 2026-08-27, not started. ⚠️ **Its pre-gate is already measured and it is NOT the cheap win it looked like**

**The change.** Lancaster `eq:pr_spitzer_adj` uses `M_sh = (4π/3)R_i³(ρ̄ − ρ_i)` — subtract the
ionised gas mass from the shell's inertia — with their own caveat that it is *"not consistent with
the derivation … \[but\] can be more accurate"*.

⚠️ **Correction recorded before any gate runs.** B11.C2's measured **+8.55% / +9.22%** in `R2(t=1.5)`
debited **`M_cav`** — the mass C3a's *cavity* premise implies. That is **not** the K9 quantity. In
trinity's geometry the ionised gas is the shell's own inner layer, so K9 debits **the shell's
ionised layer mass**, which is a different number. **B11.C2's figure does not transfer to K9.**

**G15.0 — ionised MASS fraction (BLOCKING pre-gate) — MEASURED 2026-08-27, and it is the finding.**
Reproduce: `python docs/dev/phii-identity/harness/ionised_mass_fraction.py <run>/B3M/ <run>/B3MW01/
--stride 6 --out docs/dev/phii-identity/data/b15_ionised_mass_fraction.csv`. B3M:

| phase | `m_ion/m_profile` | median | `m_profile/shell_mass` |
|---|---|---|---|
| energy | 1.0000–1.0000 | **1.0000** | 0.249–1.961 (unreliable here) |
| implicit | 0.0070–1.0000 | 0.0322 | 0.113–1.026 |
| transition | 0.0082–0.1028 | 0.0673 | 1.000–1.002 |
| momentum | 0.1094–1.0000 | **0.4611** | 1.000–1.002 |

And `B3MW01` (`Lw × 0.1`), added when the harness was written: energy 1.0000, implicit median
0.1195, transition median 0.0988, **momentum median 0.1494** (0.1312–0.2819). So the fraction is
**strongly regime-dependent** — 0.461 at nominal wind, 0.149 at low wind — and K9's magnitude cannot
be quoted from one config. Across both runs **31 of 72 replayed rows** sit at `m_ion/m_prof ≥ 0.95`,
in energy, implicit **and** momentum.

Three consequences, all of which reshape K9:
1. **The mass fraction is nothing like the thickness fraction.** B11.0 measured `dR_ion/dR_full` =
   0.9954 in momentum; the *mass* fraction is median **0.461**. The density profile rises steeply
   outward, so most of the shell's mass sits in the thin neutral outer part. Any reasoning from
   thickness to mass is wrong.
2. **It reaches 1.0000 on real rows** — in the energy phase always, and on some momentum rows.
   Debiting there leaves a **massless shell**, and `vd = F/M` diverges. K9 is a numerical hazard,
   not a small correction.
3. **The profile integral only reproduces `shell_mass` in transition and momentum** (1.000–1.002);
   in energy/implicit it is off by up to 2×, so the fraction is untrustworthy there and those
   phases need a different measurement before K9 can even be posed.

**Gates.**
- **G15.1 — definition pinned.** K9 debits `m_ion` = the shipped profile's ionised-layer mass, NOT
  `M_cav`. Any result quoted against B11.C2's +8.55/+9.22% must say which mass it used.
- **G15.2 — floor / admissibility (BLOCKING).** `M_eff/M_shell ≥ 0.05` on **100%** of rows in every
  phase K9 is applied to. *Falsifier:* any row below ⇒ that phase scope is inadmissible without a
  different shell definition. Given G15.0, the pre-registered expectation is that **energy and
  implicit FAIL this** and K9 is momentum(+transition)-only at best.
- **G15.3 — the consistency question K9 cannot dodge.** If the ionised layer is debited from the
  shell, it is *interior* gas transmitting pressure — so the drive should act at `R_IF`, on the
  neutral shell, not at `R2`. Debiting the mass while still driving at `R2` with `P_HII + P_ram`
  keeps the force ledger and mass ledger inconsistent in the *opposite* direction to today.
  **Deliverable: state which geometry K9 assumes, before implementing.** This is the same
  one-defect-two-sides finding recorded for `F_rad`, and it may make K9 inseparable from K5.
- **G15.4 — dynamics with a blocking control.** Offline re-integration (pattern:
  `harness/mass_ledger_dynamics.py`); the undebited control must reproduce the run's own `R2(t_end)`
  to **≤2%** or the debited result is **VOID**. Report both gravity treatments (inertia-only, and
  inertia+gravity) — they bracket how the ionised gas is treated gravitationally.
- **G15.5 — full-run equivalence**, same ladder as G14.4, run **separately from K5**. K5 lowers the
  drive and K9 lowers the inertia — opposite signs in `R2(t)` — so a combined arm could show a null
  while hiding two real changes.

**Ordering note.** G15.0 already argues K9 is **not** the cheap independent win it appeared to be:
its admissible phase scope is narrow, its magnitude is unmeasured for the right quantity, and G15.3
may couple it to K5. K5 is the cleaner first move.

### Batch 16 — K10 composition mapping — Status: ✅ **DONE 2026-08-28 — G16.0/G16.1/G16.2 PASS; G16.3 established the signature change it predicted. The Batch 14 composition defect is SOLVED**

**Why this batch exists.** Maintainer ruled 2026-08-28: **keep ONE radius.** That drops K8 to a
future paper, leaves `shell_structure.py` untouched (its quasi-hydrostatic BC is Rahner+2017's own
and is not the defect), and makes **K10 the live candidate** — it is the one-radius reduction of
Lancaster+2025, computing `R_i` algebraically rather than tracking it as state. Batch 14's §9 entry
found that K10's advertised "one helper, zero `P_drive` edits" is true in **momentum only**, and
that Batch 13 could not have caught it because its screen computed `P_conf·(R_i/R2)²` directly
instead of routing a helper return through each phase's real `P_drive` expression. **This batch
fixes the mapping and gates it through the real expressions. Composition only — dust is Batch 17.**

**The mapping under test** (derived 2026-08-28; `ρ ≡ (R_i/R2)² ≥ 1`). The helper must return the
CEM drive minus whatever the phase's own composition already contributes:

| phase | live composition | required return |
|---|---|---|
| energy / implicit | `max(P_conf, P_HII)` | `P_conf·ρ` (**full**) |
| transition | `max(P_conf, P_HII + P_ram)` | `P_conf·ρ − P_ram` |
| momentum | `P_HII + P_ram` (`P_conf = P_ram`) | `P_ram·(ρ − 1)` (**excess**) |

One rule: `return = P_conf·ρ − (P_ram if this phase's composition adds it else 0)`. The momentum row
is that rule's `P_conf = P_ram` limit, so transition → momentum stays continuous by construction.

**Conventions pinned** (a convention mismatch is how G13.3 failed): `n_H0 = (μ_i/μ_c)·P_conf/(k_B T)`;
`R_i³ = R2³ + 3·Q_eff/(4π χ_e α_B n_H0²)` with `χ_e` explicit (`R_ch`(trinity) = `χ_e`·`R_ch`(Lancaster));
exact spherical volumes; `Q_eff` run for BOTH Batch 13 variants (A = `Qi·f_abs`, B = with dust) —
the mapping algebra is independent of `Q_eff`, so a gate that passes for only one variant is a bug
in the gate.

**Gates.**
- **G16.0 — the mapping reproduces the CEM drive THROUGH the real expressions (BLOCKING).** For
  every committed driving row, evaluate that phase's actual `P_drive` expression with the mapped
  return and require `|P_drive/(P_conf·ρ) − 1| ≤ 1e-12`. *Falsifier:* any row worse. This is
  precisely the check Batch 13 skipped.
- **G16.1 — admissibility.** The mapped return must be `≥ 0` on every row; a negative `P_HII` would
  silently subtract force. Report the minimum per phase. *Falsifier:* any negative ⇒ that phase's
  mapping is inadmissible and needs a floor **decision**, which is physics, not a code fix.
- **G16.2 — the confined limit survives the composition.** The defect Batch 14 found was the `max`
  swallowing a small excess. Require that on rows with `ρ > 1` the composed drive is strictly
  greater than `P_conf` (i.e. Lancaster's first-order term is *delivered*, not discarded).
  *Falsifier:* any row where composed drive `== P_conf` while `ρ > 1`.
- **G16.3 — D-ramp respected (BLOCKING).** In energy/implicit, `P_conf` MUST be the **ramped**
  `press_bubble`, not the un-ramped `Pb`; using the un-ramped value re-admits the defect class C3c
  removed. `press_bubble` is recoverable from committed data on confined energy rows, where
  `P_drive = max(press_bubble, 0) = press_bubble`. Report `P_conf·ρ` under both choices and the
  divergence inside the ramp window. **Pre-registered design consequence:** `params` carries
  `current_phase` but **not** `press_bubble` (verified 2026-08-28), so K10 cannot read the ramped
  pressure through the existing signature — expect this gate to establish that K10 needs
  `press_bubble` passed in. Recording that as a **signature change**, not a failure.
- **G16.4 — magnitude (measurement, no bar).** Composed drive vs shipped, per phase per config.

**Out of scope, stated so it is not read as covered:** dust (Batch 13's G13.4 fired at 2.05× and is
Batch 17), any `trinity/` edit (ship-hold), and any claim about trajectories — this is offline on
committed rows, so it cannot say what K10 does to a run it drives itself.

---

#### Batch 16 RESULT — 2026-08-28, measured against the gates above

Reproduce: `python docs/dev/phii-identity/harness/k10_composition_check.py --out
docs/dev/phii-identity/data/b16_composition.csv`. No solver run; 853 (row × variant)
evaluations on the committed c3c-arm trajectories, both `Q_eff` variants.

- **G16.0 ✅ PASS — worst rel err 2.22e-16 against a 1e-12 bar.** The mapping reproduces
  `P_conf·ρ` through each phase's **real** `P_drive` expression: energy 0.00e+00 (n=312),
  implicit 0.00e+00 (n=296), transition 2.22e-16 (n=144), momentum 2.22e-16 (n=101), and it
  holds for **both** `Q_eff` variants as required. **The Batch 14 composition defect is solved:
  one rule — `return = P_conf·ρ − (P_ram if the phase's composition adds it)` — makes all three
  existing compositions yield the CEM drive exactly.**
- **G16.1 ✅ PASS — 853/853 returns ≥ 0**, minimum comfortably positive in every phase
  (momentum +1.22e2 the tightest). No floor decision needed, which was the registered risk.
- **G16.2 ✅ PASS — 853/853.** The confined-limit term is **delivered, not swallowed**: median
  ED-phase excess over `P_conf` is **+0.96%**, where the shipped helper returns exactly 0.0.
  This is Lancaster's first-order term actually reaching the drive — the specific failure
  Batch 14 identified in the excess-only form.
- **G16.3 — the pre-registered design consequence is confirmed, and it is not optional.**
  Inside `dt_switchon`, `P_conf_ramped/P_conf_unramped` = **0.3302–0.9952 (median 0.7112)** over
  264 rows, so using the un-ramped `Pb` would overstate the confining pressure by up to **3×**
  exactly inside the window C3c's D-ramp fix protects. Since `params` carries `current_phase`
  but **not** `press_bubble`, **K10 requires the ramped pressure to be passed in — a signature
  change to `get_phii_c3c`.** Recorded as a design consequence, as registered, not a failure.
- **G16.4 — magnitude, and an independent cross-check of Batch 13.** Composed/shipped medians
  reproduce Batch 13's numbers **exactly** through a completely different code path (B3M
  momentum 1.605 A / 0.851 B; B3MW01 momentum 2.242 A / 1.096 B; transition 1.381/0.794 and
  2.074/1.059). The healthy branch is barely touched (B3M energy 1.005, implicit 1.007),
  consistent with G13.2's +0.68%. B3MW01's ED phases are the divergent ones (1.137/1.225).

**Caveats, stated.** (i) `press_bubble` is recovered from `P_drive` on confined rows, so it
inherits the documented stale-`Pb` handoff rows — `ramped/unramped` reaches 1.0732 *outside*
the ramp window on a few energy rows where it should be exactly 1. Harmless to G16.0 (which is
self-consistent per row) but it means the recovered `press_bubble` is not exact on those rows.
(ii) **Dust coverage is sparse in the ED phases** (B3MW01 implicit n=1) because the photon
ledgers only replayed driving rows — the B-variant ED numbers are not a sample worth quoting.
(iii) Offline on committed rows: says nothing about a trajectory K10 drives itself.

**What this unblocks and what it does not.** The composition question is closed. K10's remaining
blockers are now exactly two, both pre-existing: **the dust model** (G13.4 fired at 2.05×;
Batch 17) and **a full-run arm**. The signature change is small and additive
(`get_phii_c3c(params, shell_props, P_conf=None)`), but it is still a `trinity/` edit and
therefore still behind the ship-hold.

---

### Batch 17 — dust inside the K10 closure — Status: ✅ **DONE 2026-08-28 — G17.0/G17.1/G17.2 PASS; G17.3's pre-registered expectation MISSED (recorded, diagnosed). G13.4's blocker is discharged: K10 now has a dust model validated against the code's own**

**Why this batch exists.** Batch 13's **G13.4 fired at 2.05×** — the drive's sensitivity to whether
dust is in the photon budget exceeded the pre-registered 2× bar — and the verdict as registered was
**"K10 cannot ship without a dust model."** Batch 13's dust was a *post-hoc join*: `f_dust` read from
the photon ledgers at nearest `t`, which is a table lookup, not a closure. Batch 16 closed the
composition question, so dust is the last piece that can be settled offline. Batch 16's mapping is
used end-to-end here, so this batch measures the **complete** candidate.

**The closure under test — a reduction of trinity's OWN dust physics, not a new model.**
`get_shellODE.py:120` is the code's ionised-region photon equation:

> `dφ/dr = − 4π r² χ_e α_B n² / Qi − n σ_d φ`

with `σ_d = params['dust_sigma']`. Two sinks: recombination and dust. K10's approximation is a
**uniform** layer density `n₀ = (μ_i/μ_c)·P_conf/(k_B T)` (identically `shell_structure.py:125`'s
`nShell0`), so the closure is that same ODE integrated at constant `n₀` from `R2` with `φ(R2) = 1`,
and the ionisation front is `R_i := r` where `φ = 0`. The drive is `P_conf·(R_i/R2)²`, composed via
Batch 16's mapping. **`Qi` is used whole, not `Qi·f_abs`** — the shell solve starts at `φ = 1` with
the full budget (§6b item 5, B11.0 seam A), and the recombination/dust/escape split is an *output*
of the solve rather than an input. This deliberately differs from Batch 13's `Q_eff` and is the
point of the batch.

**Conventions pinned:** `χ_e` explicit; exact spherical volumes; code units throughout
(`σ_d` [pc²], `n` [pc⁻³], `α_B` [pc³/Myr], `Qi` [1/Myr]) so `nσ_d` and the recombination term are
both [1/pc]; `P_conf` is the **ramped** `press_bubble` in energy/implicit per **G16.3**.

**Gates.**
- **G17.0 — the closure's dust fraction reproduces the shell solve's own (BLOCKING).** The closure
  predicts `f_dust` = the fraction of `Qi` absorbed by dust, `= ∫ n₀ σ_d φ dr` over the layer.
  Compare against the run's own measured value (`f_ionised_dust` in `data/b9_layer_density.csv`;
  `dust_Pb` in the photon ledgers) on matched rows. **Bar: median predicted/measured ∈ [0.5, 2.0]**
  (factor 2), and additionally report the fraction of rows within 25%. *Falsifier:* median outside
  that band ⇒ the uniform-density reduction does **not** capture trinity's dust, and K10 needs the
  real profile rather than a closure. ⚠️ Disclosed prior: G9.4 measured the analytic uniform form
  overstating the profile's recombination-equivalent density by up to **3.17×**, so this gate is
  genuinely at risk and a failure is a real possible outcome, not a formality.
- **G17.1 — convergence.** The front solve must find a bracketed `φ = 0` root on 100% of rows
  attempted. *Falsifier:* any non-convergent row ⇒ report it and its cause; a row where `φ` never
  reaches 0 means no front exists at uniform density (the shell is photon-leaking there) and is
  reported as `no_front`, **not** silently dropped.
- **G17.2 — the `σ_d → 0` limit recovers Batch 13's variant A exactly.** With `σ_d = 0` the closure
  must reproduce `R_i³ = R2³ + 3·Qi/(4π χ_e α_B n₀²)` to **1e-10** relative. *Falsifier:* worse ⇒
  algebra or units bug, and nothing else in the batch is trustworthy.
- **G17.3 — G13.4's sensitivity is now internal.** Report the drive under (a) no dust, (b) Batch
  13's post-hoc `Q_eff = Qi·f_abs·(1−f_dust)`, (c) this self-consistent closure. **Pre-registered
  expectation, disclosed: (c) lands between (a) and (b), nearer (b).** Re-run G13.4's ratio with (c)
  in place of the join and report whether it still exceeds 2×.
- **G17.4 — end-to-end magnitude (measurement, no bar).** Composed drive (Batch 16 mapping) over
  shipped drive, per phase per config, using the self-consistent closure.

**Out of scope:** any `trinity/` edit (ship-hold); trajectories (offline on committed rows only);
the `f_esc > 0` regime is reported, not modelled.

---

#### Batch 17 RESULT — 2026-08-28, measured against the gates above

Reproduce: `python docs/dev/phii-identity/harness/k10_dust_closure.py --out
docs/dev/phii-identity/data/b17_dust_closure.csv`. No solver run; 436 rows over both configs.
`σ_d` = 1.5754e-58 pc² (the code's own `dust_sigma`).

- **G17.0 ✅ PASS, and not marginally — the closure's dust IS trinity's dust.** Predicted/measured
  `f_dust`: **median 1.056** (range 0.534–1.120) against a [0.5, 2.0] bar, with **97.3% of rows
  within 25%**, and the two configs agreeing independently (B3M 1.064 over 61 rows, B3MW01 1.052
  over 52). Absolute values: closure median **0.6629** vs measured **0.6215**.
  ⚠️ **The disclosed prior risk did not materialise, and it is worth saying why.** G9.4 measured the
  uniform analytic form overstating the profile's *recombination*-equivalent density by up to 3.17×,
  so this gate was registered as genuinely at risk. It passes because dust absorption depends on
  `∫ n σ_d φ dr` — **linear** in `n` — while recombination goes as `n²`, so the dust fraction is far
  less sensitive to the profile shape than the recombination integral that G9.4 was measuring. The
  uniform-density reduction is a poor model of the *density* and a good model of the *dust*.
- **G17.1 ✅ PASS — 436/436 fronts converged**, no `no_front` rows.
- **G17.2 ✅ PASS — worst rel err 9.26e-13** against a 1e-10 bar: with `σ_d = 0` the integration
  recovers `R_i³ = R2³ + 3Qi/(4π χ_e α_B n₀²)` exactly, so the algebra and the code-unit chain are
  verified end to end.
- **⛔ G17.3 — MY PRE-REGISTERED EXPECTATION MISSED. Recorded as a miss, not reinterpreted.** I
  registered that the self-consistent drive **(c)** would "land between (a) no-dust and (b) Batch
  13's post-hoc, nearer (b)". It does **not**: on 4 of 5 phase×config groups **c < b < a**, i.e. c
  sits just *outside* the bracket, below the post-hoc value (`c/b` = 0.92–0.95). Only B3MW01's
  single implicit row is contained. **The direction was right and the containment claim was wrong.**
  Diagnosed: the post-hoc form applies `(1 − f_dust)` **once** to the photon budget, whereas the
  closure removes photons **continuously along the layer**, where dust competes with recombination
  at every radius — so it limits `R_i` slightly more than a single up-front debit.
  **G13.4 re-run with (c) in place of the join:** no-dust/self-consistent = **1.938 / 1.886**
  (B3M transition / momentum) and **2.214 / 2.184** (B3MW01). So the sensitivity is **still ≈2× and
  still above the old bar on B3MW01** — dust genuinely matters this much. That does **not** revive
  the blocker: G13.4's verdict was *"K10 cannot ship without a dust model"*, and the deliverable was
  a model, not a smaller number. The dust fraction is now **computed inside the closure and
  validated against the code's own shell solve to 5.6%** (G17.0), so it is no longer a free choice.
- **G17.4 — end-to-end magnitude** (dust closure + Batch 16 mapping, composed through the real
  `P_drive` expressions), median composed/shipped:

| config | energy | implicit | transition | momentum |
|---|---|---|---|---|
| B3M | 1.006 | 1.005 | 0.746 | 0.884 |
| B3MW01 | 1.059 | 1.099 | 0.982 | 1.034 |

  The healthy branch is essentially untouched on B3M (1.005–1.006), and the whole candidate sits
  within ~25% of the shipped drive everywhere and within ~10% on B3MW01. Dust optical depth across
  the closure layer is **0.246–4.521 (median 1.591)**, i.e. the layer is marginally optically thick
  to LyC on dust — consistent with dust mattering at the measured ~2×.

**⚠️ Coverage gap, stated because G17.4 quotes numbers that depend on it.** G17.0's comparison rows
are **transition 70 / momentum 42 / implicit 1** — the photon ledgers replayed the driving branch,
so **the ED-phase dust fraction is validated on a single row.** G17.4's energy/implicit columns
therefore rest on an *unvalidated* dust fraction in those phases. Extending the ledgers to confined
rows is the cheap fix and is the natural first step of any follow-up. Two smaller conventions to
note: the ledger's `dust_Pb` is solved at `nShell0(Pb)` while the closure uses `P_conf`, which are
identical in momentum (`Pb := P_ram`) and can differ in transition when `P_ram > Pb`; and this is
offline on committed rows, so it says nothing about a trajectory K10 drives itself.

---

### Batch 18 — the K10 arm — Status: ⛔ **HELD 2026-08-29 by Batch 20 slice 2** (seam C measured PRESENT and worse than C3c's; drive and mass inconsistency are the same number). Per-call work stands: G18.0 FAILED as written (diagnosed), G18.0′ PASSES at 1.005e-12, arm runs clean. **SUPERSEDED 2026-08-29 by Batch 21 (`hpc/b14/k10_o1_arm.patch`)**, which disposes of slice 2 by changing the front source rather than by clearing the Batch 18 form: G21.1/G21.2 measured the domain violation and seam C both closed. **This arm is HELD permanently; the ladder is owed against the O1 patch.** `run_arms.sh` now refuses `k10` unless explicitly forced.

**Authorisation.** Maintainer, 2026-08-28: **ship-hold lifted for the K10 arm only.** Scope, as
granted: implement K10 and measure what it does to a trajectory it drives itself. **NOT** granted: a
default flip, D5 adoption, or any claim that K10 ships. D5 stays **open** by the maintainer's
explicit ruling the same day ("not yet — wait for the arm"), so §7.1 is unchanged.

⛔ **G16.3's "signature change" consequence is RETRACTED (2026-08-28).** Batch 16 concluded K10 would
need `get_phii_c3c(params, shell_props, P_conf=None)` because `params` carries `current_phase` but
not `press_bubble`. That inference was wrong: **`get_effective_bubble_pressure` lives in the same
module** (`get_bubbleParams.py`) and every one of its inputs — `current_phase`, `Eb`, `R2`, `R1`,
`gamma_adia`, `Lmech_total`, `v_mech_total`, `t_now`, `tSF` — **is in `params`**. The helper can
therefore compute the ramped confining pressure itself. **No signature change; no call-site edits.**
The G16.3 *measurement* (ramped/un-ramped 0.3302–0.9952 inside `dt_switchon`) stands and is why the
ramped value must be used; only the implementation consequence was wrong.

**What the arm implements** — `get_phii_k10`, self-contained, one function:
1. `P_conf = get_effective_bubble_pressure(...)` from `params` — the **ramped** value (G16.3).
2. `n₀ = (μ_i/μ_c)·P_conf/(k_B T)` — identically `shell_structure.py:125`'s `nShell0`.
3. `R_i` from the **dusty** front (Batch 17): the closed-form solution of
   `dφ/dr = −4πr²χ_e α_B n₀²/Qi − n₀σ_d φ`, `φ(R2) = 1`, root-found on the guaranteed bracket
   `[R2, R_i_nodust]`.
4. Return **Batch 16's mapping**: `P_conf·ρ − (P_ram if this phase's composition adds it)`, with
   `P_ram` computed directly via `pRam(R2, Lmech_total, v_mech_total)` rather than read from
   `params['P_ram']`, so the helper carries no call-ordering dependence.

**Gates.**
- **G18.0 — per-call equivalence to the screened closure (BLOCKING, runs first).** The implemented
  helper, driven on the committed rows, must reproduce `data/b17_dust_closure.csv`'s
  `drive_selfconsistent` and `composed_selfconsistent` to **1e-10** relative. *Falsifier:* any row
  worse ⇒ the production code is not the thing Batches 16/17 validated, and the arm is void. This is
  CLAUDE.md rule 5's "per-call first, necessary but not sufficient".
- **G18.1 — confined-branch contract change is EXPLICIT.** C3c returns exactly 0.0 while confined and
  `test/test_phii_c3c.py` pins it; K10 deliberately returns a small positive excess there (Batch 16
  measured +0.96% median). The arm therefore **breaks those pins by design**. *Deliverable:* a
  before/after table in the D4 style, and the pinning tests are NOT re-baselined in this arm — a
  broken pin is recorded, not silenced.
- **G18.2 — numerical health.** Zero excess-work / overflow / monotonic-guard / convergence warnings
  across the ladder; wall times within 2× of the shipped baseline. *Falsifier:* distress lines ⇒
  report the regime and stop.
- **G18.3 — full-run equivalence (CLAUDE.md rule 5).** Separate processes, matched `t`, on
  `SC + F1LO + F1HI + B3M + B3MW01` against a shipped baseline at the same SHA. Fate table
  before/after; ΔR2 reported, **no bar** — K10 is expected to move the trajectory (Batch 17 put the
  drive within ~25%), so this is a measurement, and a *fate flip* is the reportable event (D3).
- **G18.4 — the continuity claim, on a trajectory K10 drove.** The shipped rule jumps **+34.0%** at
  the branch switch (Batch 13, at fixed state). K10 has no branch, so its state-jump is 0 by
  construction — but on its OWN trajectory the test is whether the `t_cross` kink and the
  §3c.1 seam ratios are smoother than the shipped arm's. Report both arms' adjacent-row drive ratios
  at the seams. *No bar* — the shipped comparator is the reference.
- **G18.5 — does the ED-phase dust gap bite?** Batch 17 validated ED dust on **one row**. The arm
  drives the energy phase with that unvalidated fraction. Report the run's own
  `shell_fIonisedDust` against the closure's predicted `f_dust` on ED rows — this **closes G17.0's
  coverage gap using the arm's own output**, at no extra cost.

**Out of scope:** default flip, D5 adoption, `phii_scheme` key, golden re-baselining.

---

#### Batch 18 RESULT (per-call half) — 2026-08-28, measured against the gates above

Implemented as an **arm patch**, never on `main`: `docs/dev/phii-identity/hpc/b14/k10_arm.patch`
adds `get_phii_k10` + `_k10_front_radius` and aliases `get_phii_c3c` to it, so **zero call sites
change** and reverting is deleting one line. `scipy.optimize` was already imported; no new
dependency. Reproduce by applying the patch in a detached worktree (`run_arms.sh prep <sweep> k10`).

- **⛔ G18.0 — FAILED AS WRITTEN. Recorded, not re-barred.** Worst rel err **6.761e-02** against a
  1e-10 bar. The failure is *entirely* the `P_conf` **source**, and the per-phase pattern proves it:
  `P_conf` rel err is **0.000e+00 in implicit and momentum** and **6.817e-02 (energy) / 5.877e-03
  (transition)**. Batches 16/17 *recovered* `P_conf` from stored columns (`P_drive` on confined rows,
  stored `Pb` elsewhere); production *recomputes* it from `Eb` and a freshly solved `R1`. The two
  agree exactly wherever recomputation is trivial and diverge where it is not — inside the
  `dt_switchon` ramp window (energy) and through the `Eb → Pb` reconstruction (transition).
  **Production is the correct side; the offline screen was the approximation** — which Batch 17's
  caveat had already flagged qualitatively ("the recovered `press_bubble` is not exact on those
  rows"). The gate did its job: it caught a real screen-vs-production difference that would
  otherwise have been carried into the arm silently.
- **✅ G18.0′ (amendment, following this workstream's own G8.4′ precedent) — PASSES at 1.005e-12.**
  Same 1e-10 bar on the same quantity, with the one input the screen could not reproduce held fixed:
  feed production's `_k10_front_radius` the screen's `P_conf` and compare drives. Per phase: energy
  1.005e-12, implicit 3.708e-14, transition 2.511e-13, momentum 3.095e-13. **So the production
  closure IS the object Batches 16 and 17 validated**; only the confining-pressure source differed.
  The amendment does not weaken the closure test — it isolates it.
- **✅ Arm runs clean (partial G18.2).** `SC` to `stop_t` = 0.01: **ok, 114 snapshots, 338 s, zero
  distress lines** (no excess-work / overflow / monotonic-guard / convergence warnings).
- **✅ G18.1's contract change is confirmed live and is exactly as designed.** On that run
  `P_HII > 0` on **97/97 energy rows** and **17/17 implicit rows**, where C3c returns exactly 0.0 on
  100% of them. Median `P_HII/Pb` = 0.8392 (energy — the ramped `P_conf` sits below the stored
  un-ramped `Pb`, so this is the D-ramp being honoured) and 1.0008 (implicit, i.e. the excess is
  +0.08% there). `test/test_phii_c3c.py` fails under the arm **by design**; the pins are recorded
  broken, not re-baselined.

⚠️ **Consequence for Batch 17, quantified and limited.** G18.0's `P_conf` diagnostic bounds the
error in Batch 17's offline magnitudes: **momentum and implicit are exact (0.0)**, **transition
carries ≤0.59%**, and energy carries ≤6.8% **on 2 of 156 rows only** (⛔ narrowed
2026-08-29 from a blanket "energy ≤6.8%": both are the known `t` = 3.0e-3 stale-`Pb` 1a→1b handoff,
plus one B3MW01 transition duplicate at `t` = 0.844617). So §Batch 17's G17.4 energy column is sound
apart from those rows. G17.0's dust validation is unaffected — its
comparison rows are transition/momentum/implicit only.

⚠️ **Environment note (not an arm defect).** The local interpreter is **Python 3.8.8**, below
trinity's stated **≥3.9** minimum, and `harness/run_batch.py` uses `Path.is_relative_to` (3.9+), so
its post-run *reporting* raises after the run completes and the walltimes CSV is skipped. Run
outputs are unaffected (written before the raise). On Helix `sync.sh` activates the `trinity` conda
env, so this should not bite there — but it is worth fixing or noting before the ladder.

**Still owed for this batch:** G18.2 (full ladder health), **G18.3 (full-run equivalence, the
sufficient half of CLAUDE.md rule 5)**, G18.4 (continuity on K10's own trajectory), G18.5 (ED-phase
dust, which the arm's own output closes for free). All need the ladder:
`docs/dev/phii-identity/hpc/b14/sync.sh` now defaults to `ARMS="baseline k10"`.

---

### Batch 20 — K10 safety audit — Status: ⛔ **VERDICT 2026-08-29: K10 is UNSAFE as implemented — 4 MAJOR + 1 MODERATE (the photo-only-limit finding was re-labelled from CRITICAL-runtime to MAJOR-domain on 2026-08-29, and slice 1 later added a 4th MAJOR) (seam C worse than C3c; per-segment freeze ratchet re-armed; coverage 2 configs of 13). Arm HELD.** All four slices in. Slice 1 additionally found a float64 bracket failure reachable at low metallicity, and a test (`test_mu_audit_drift.py`) that now passes vacuously; it also **refuted my own registered suspicion** about the `n_IF_Str` gate (0/3490 rows).

**Why.** Maintainer asked, after Batch 13's cancellation claim was found false: *"can you check other
claims too so i know K10 is safe or unsafe?"* The cancellation claim was checkable by algebra and had
stood for two days on plausibility alone. That is a reason to distrust the **other** load-bearing
claims in the K10 case rather than only the one that broke. This section is written and committed
**before any slice reports**, so no finding can be retro-fitted to the questions.

**Method.** Four independent read-only audits, run in parallel, each told to try to REFUTE rather
than confirm, and each restricted to committed artifacts + source (no `trinity/` edits, no commits).
The K10 case as it stands rests on the claims below; each slice owns one group.

**Slice 1 — implementation safety** (the arm patch's numerics). Pre-registered targets: the
`k·(hi−R2) < 1e-8` thin-dust guard and whether the closed form suffers catastrophic cancellation
just above it (`P(s) = s²/k − 2s/k² + 2/k³` is a cancellation-prone form for small `k`); whether
`φ(hi) < 0` is genuinely guaranteed so `brentq` always brackets; whether the **transition** mapping
`P_conf·ρ − P_ram` can return a NEGATIVE pressure (which would subtract force); degenerate inputs
(`Qi→0`, `P_conf→0`, `R2→0`, `Eb→0`); float64 headroom given `n₀ ~ 1e64` and `A ∝ n₀² ~ 1e128`;
per-call cost in the hot loop. **Plus one specific suspected defect I want checked, stated here so
the answer is on the record either way:** `get_phii_k10` ignores `shell_props` entirely, but all six
call sites are guarded by `if params['include_PHII'].value and n_IF_Str > 0:` — so K10's value is
**discarded whenever `n_IF_Str == 0`**, a gate it neither sets nor controls.

**Slice 2 — are seams A and C really "absent by construction"?** The phrase appears throughout §7.1
and §Batch 13 and has never been checked. Seam A: K10 uses the **full `Qi`** with `φ(R2) = 1`, and so
does the shell solve — is that the same photons counted once in two consistent ways, or a second
sink? Seam C: compute the gas mass K10's own `n₀` implies over its layer `(4/3)π(R_i³ − R2³)` and
compare with the run's `shell_mass`. **If the implied layer mass exceeds the shell's own mass, seam C
is NOT absent and the claim is false.** Also asked: does K10 introduce a *new* seam of the §6b seam-B
type, since its uniform `n₀` and the shell solve's real profile describe the same gas?

**Slice 3 — the limits and the Lancaster mapping**, each to be re-derived independently rather than
restated: (a) the confined limit really giving Lancaster's first-order `(2/3)(R2/R_ch)·P_conf`;
(b) whether K10 recovers the **Spitzer / D-type** limit — load-bearing, because Batch 8 established
that limit as *the one exact external anchor* the scheme family has, and if K10 loses it the
strongest external validation goes with it; (c) re-verification of G13.3's `R_ch`(trinity) =
`χ_e`·`R_ch`(Lancaster) diagnosis and its `χ_e^{2/3} − 1` error prediction; (d) whether K10 literally
contains K5's volume fix; (e) a full code-unit dimensional check of the photon ODE.

**Slice 4 — regime coverage and failure modes.** All K10 screening used **two configs**, B3M and
B3MW01. Registered questions: which of the workstream's matrix regimes have never been screened;
outliers in `ρ` and `τ_dust` in the committed screens; the **photon-leaking regime** (`f_esc > 0`,
where no ionisation front exists and K10's bracket premise may fail); whether `P_conf` itself injects
a discontinuity at the transition→momentum handover even though K10 has no internal branch; and an
explicit statement of the evidence gap against C3c's 13-config clean record.

**What this batch cannot do.** It is offline and adversarial only. It cannot establish that K10 is
*correct* — only whether the stated case for it survives scrutiny. Correctness needs G18.3.

**Bar.** Any CRITICAL or MAJOR finding blocks the arm until dispositioned. Findings that merely
correct the *record* (as Batch 13's did) do not block, but must land in the doc dated.

#### Batch 20 VERDICT — 2026-08-29: **K10 is UNSAFE as implemented.** 4 MAJOR + 1 MODERATE after the 2026-08-29 re-label and slice 1's additions (the heading's original "1 CRITICAL + 3 MAJOR" is superseded), arm stays HELD

All four slices reported. ⚠️ This VERDICT was written when three were in; slice 1's additions (F2 CRITICAL-where-reachable, F8 MAJOR review-hygiene) are recorded in §Slice 1 RESULT below but are NOT folded into the headline counts in this section's heading — see the 2026-08-29 housekeeping entry in §9. Every finding below
that I could check independently, I did — those are marked ✔ verified.

**⛔ CRITICAL — K10 breaks the photo-only (Spitzer / D-type) limit, silently.** ⚠️ **RE-LABELLED 2026-08-29 to MAJOR-domain, not CRITICAL-runtime** — `P_conf = 0` is unreachable in a real trinity run, so the fixture is artificial *as a runtime hazard*. The finding survives in a stronger form (the front is outside the cloud on 100% of driving rows); see the follow-up section below. ✔ **Verified by me
directly**, applying `hpc/b14/k10_arm.patch` in a clean worktree and running the real suite:
`test/test_phii_c3c_spitzer.py` goes **6 passed (shipped) → 5 failed, 1 passed (K10)**. Mechanism:
Batch 8's fixture turns the wind off (`Pb = 0, Eb = 0, Lmech = 0`), so `P_conf = 0`, the guard
`if not (P_conf > 0.0): return 0.0` fires, and K10 returns **exactly 0.0 photoionised pressure at
every radius** where C3a returns the classical D-type pressure. The measured expansion index is
`−1.65e-16` instead of 4/7. **This is structural, not a bug:** `drive = P_conf·(R_i/R2)²` with
`R_i ∝ n₀^{−2/3} ∝ P_conf^{−2/3}` gives `drive ∝ P_conf^{−1/3}` — the limit is **singular**, and the
guard converts a divergence into zero. ✔ **The divergence is already visible in the committed data**:
momentum `drive/P_conf` is 6.213 (B3M) vs 15.265 (B3MW01, `Lw`×0.1), a ratio of **2.457** against the
`10^{1/3} = 2.154` the scaling predicts — 14% agreement over one decade of wind. **So K10's drive
grows without bound as the wind weakens, and it has no photoionisation-only limit at all.** Batch 8
established that limit as *the one exact external anchor* this scheme family has. ⚠️ **And this is my
own gate-design failure**: Batch 14's **G14.2** explicitly protected exactly this for K5 ("the
photo-only limit is the one anchor C3a has"); I did not carry a limits gate into Batch 18, so
G18.0–G18.5 could never have caught it.

**⛔ MAJOR — seam C is present and WORSE than C3c's; see §Slice 2 RESULT below.** ✔ Headline verified
(253,412 vs 101,805 Msun, ratio 2.4892). The blanket "all four seams absent by construction" is
withdrawn for K10-as-implemented.

**⛔ MAJOR — K10 re-arms the per-segment freeze ratchet that C3c disarmed.** ✔ **Verified in source.**
`run_energy_phase.py:230` writes `params['P_HII']` once per outer step and it is frozen into the ODE
snapshot; `energy_phase_ODEs.py:249` reads `snapshot.P_HII` — **constant across the whole `solve_ivp`
segment** — while `press_bubble` at `:224` is recomputed **live** from the integrating `(Eb, R2, R1,
t)`. The composition is `max(press_bubble_live, P_HII_frozen)`. Under C3c `P_HII ≡ 0.0` on every
confined row, so the frozen term is inert. Under K10 it is `P_conf(t_k)·ρ > press_bubble` for the
rest of the segment whenever `Pb` falls by more than `ρ−1`:

| config | phase | within-segment `Pb` decline | K10's `ρ−1` |
|---|---|---|---|
| B3M | energy | median **8.03%**, max **16.75%** | median **0.554%** |
| B3M | implicit | median 5.34%, max 8.42% | 0.527% |
| B3MW01 | energy | median 8.74%, max 17.02% | 5.86% |

So a smoothly declining drive becomes a **piecewise-constant staircase with ~8% median and up to 17%
steps — 15× larger than the +0.55% Lancaster term K10 exists to deliver.** This workstream already
condemned this defect class: **§3's C2a row rates it "catastrophic at compact scale"** (phase1a-init
Extra finding #1), and it is the *same* freeze the pre-C3c `P_HII` had. K10 respects the ramp (G16.3)
and re-introduces the freeze. Momentum is not a regression (C3c's `P_HII` is equally frozen there);
this is ED-phase-specific. **The compact config PRB is exactly the "catastrophic at compact scale"
case and has never been run.**

**⛔ MAJOR — coverage is far narrower than the case implies.** Every K10 artifact derives from **two
trajectories of the SAME cloud** (`bench3_m1e5_r5`) at two wind strengths, to 1.5 Myr. Never
screened: **`F1LO, F1HI, PRB, WW, B1M, B2M, GMC, BE, PL2, LDLS, SDHS, B3MW3, B3MW10, B3MW001,
B3ML`** — 4 decades of density, 4.5 of mass, both non-power-law profiles, **all three collapse-fate
configs**, and everything past 1.5 Myr (B3M's own `stop_t` is 5; the default is 15). C3c landed on 13
configs with a full fate table. K10 has one config for 0.01 Myr.

**⚠️ MODERATE — a reachable crash band in `_k10_front_radius`.** The patch docstring claims "the root
always exists — there is no non-convergence branch". True in exact arithmetic, **false in float64**:
`P(s) = s²/k − 2s/k² + 2/k³` cancels catastrophically for small `τ = k(hi−R2)`. Measured against
60-digit reference: τ=1e-4 → root degrading; τ=1e-5 → root wrong; **τ ≤ 1e-7 → `brentq` raises
`ValueError`, unhandled, taking the run out.** The guard is at 1e-8, four orders too low. Measured τ
on the two screened configs is 0.246–4.521, which is exactly why the screen could not see it — but it
is **reachable by two matrix-legal levers**: `dust_sigma ∝ ZCloud` (a `ZCloud = 1e-3` dwarf puts B3M's
implicit τ at 2.5e-4), and late-time `Qi` fade (`Qi(15 Myr)/Qi(1.5 Myr) = 5.1e-4` from the bundled
SB99 table, putting τ ≈ 1.3e-4 at the *default* `stop_t`). **Fix is one line: raise the guard to
~1e-3**, plus a `try/except` fallback to `hi`.

**⚠️ MODERATE — K10 manufactures an ionisation front where the code says photons escape.** `f_abs < 1`
on **145/462 (31%)** of B3M rows and 127/427 of B3MW01; on the *median* B3M energy row **67% of
ionising photons escape and no front exists**. K10 uses `Qi` whole and its bracket is guaranteed, so
**it can never report "no front"** — G17.1's 436/436 convergence is not reassurance, it is proof the
`no_front` path can never fire. On B3M energy rows K10's layer is **7.9× thicker** than the shell
solve's measured `dR_ion`. Drive error there is bounded (~0.5–4%) but ungated.

**⚠️ MINOR, but fix before any arm run.** (a) `F_HII` semantics: the Batch 16 mapping makes the
return phase-dependent, so in energy/implicit the reported `F_HII = 4πR2²·P_HII` is the **entire**
drive force — any force-budget figure would read "photoionisation supplies 100% of the drive". It is
diagnostic-only (`vd` uses `P_drive`, verified), so not a dynamics defect, but it is exactly the
honesty problem §3's C4 exists to fix. (b) The patch docstring's "+0.96%" confined excess is Batch
16's **no-dust** variant number; the dusty closure the code actually runs gives **0.672%**, and the
live SC run measured +0.08%. (c) Cost is unmeasured: K10 adds **two** root-finds per call
(`solve_R1`, which C3c never calls, plus the front solve) at six sites, with no same-SHA comparator.

**✅ What survived scrutiny — and should not be re-litigated.** Seam A **is** absent as claimed (the
closure integrates literally `get_shellODE.py:120`; `f_abs` appears zero times in the patch; there is
no cavity term, so it is a reduced model *of* the shell solve's sink, not a second one). The
composition mapping is settled (**2.22e-16**, 853 evaluations, both `Q_eff` variants) and
non-negativity is not merely empirical but **provable** (`P_conf = max(P_thermal, P_ram) ≥ P_ram` and
`ρ ≥ 1`). The dust closure is genuinely validated where it was checked (median 1.056, 97.3% within
25%, two configs agreeing independently) and `σ_d → 0` recovers the analytic form to 9.26e-13.
G13.3's `R_ch(trinity) = χ_e·R_ch(Lancaster)` diagnosis is **confirmed and strengthened** — the exact
per-row form is `[(1+χ_e x)/(1+x)]^{2/3} − 1`, reproducing `b13_k10_screen.csv`'s
`md_identity_relerr` to **1.04e-14 on all 59 momentum rows**, with `χ_e^{2/3}−1` being only its
`x → ∞` asymptote. Units are consistent throughout. And the transition→momentum handover carries **no
K10-specific jump** (`P_ram/Pb` climbs smoothly to exactly 1.0000; worst step 0.24% B3M / 1.18%
B3MW01) — a registered worry that did not materialise.

**✏️ Corrections to my own record, from the audit.** (1) The G18.0 `P_conf` discrepancy is narrower
than §Batch 18 states: it is confined to **2 of 156 energy rows** (both the known `t` = 3.0e-3
stale-`Pb` 1a→1b handoff) plus one B3MW01 transition duplicate at `t` = 0.844617 — not a blanket
"energy carries ≤6.8%". (2) "K10 contains K5's volume fix by construction" is true of the *equation*
and false of the *value*: with dust the balance is over `Q(1−f_dust)`, and the plan's own 2026-08-29
retraction already shows corrected-C3a/K5b (1.545) sitting 4.10× below K10 (6.333).

**Disposition.** Batch 18 stays **⛔ HELD**. Before any ladder: fix the guard band, add a **limits
gate** to Batch 18 (the G14.2 analogue I omitted), decide whether the CEM form's missing photo-only
limit is acceptable **as physics** — it is a scope statement about the model, not a code defect — and
disposition the freeze ratchet, which may require calling the helper live inside the ODE rather than
freezing it per segment. **The cheapest real progress is not the arm**: re-point the existing offline
screeners at the other core-6 trajectories, which needs no new physics and no `trinity/` change.

#### Batch 20 follow-up — the maintainer's challenge to the CRITICAL finding, and the REFRAME it forces

**The challenge (2026-08-29):** *"what if a photoionised limit cannot be reached in our Weaver-like
wind-driven bubble?"* — i.e. `P_conf` is strictly positive in every trinity phase (thermal `Pb`,
`max(P_thermal,P_ram)`, `P_ram`), so a fixture that switches the wind off is artificial and the
broken Spitzer limit may be irrelevant.

**On the literal question the maintainer is RIGHT, and the CRITICAL finding is reframed, not
withdrawn.** `P_conf = 0` is unreachable in a real run, so `test_phii_c3c_spitzer.py`'s 5 failures
are **not** a runtime hazard. As a *runtime* severity that finding drops. What it was really
detecting is a **domain-of-validity** problem, and the right question is not "what happens at the
singularity" but "how far into the divergence do trinity's own runs already sit". Measured
(`harness/k10_domain_check.py` → `data/b20_domain.csv`; `rCloud` = **4.9990 pc**, derived from the
code's unit-converted `mCloud`/`nCore` with a coefficient of 800.465 that reproduces slice 2's
independently measured 800.5):

**D1 — K10's ionisation front sits beyond the shell the ODE actually produced:**

| B3M phase | rows beyond the shell's outer edge | median `R_i`/shell_outer |
|---|---|---|
| energy | **43/44** | 1.002 |
| implicit | 8/34 | 1.000 |
| transition | **17/21** | 1.217 |
| momentum | **18/18** | **1.281** |

**D2 — and beyond the cloud itself, on every single driving row:**

| config | phase | beyond `rCloud` | median `R_i`/`rCloud` | max `R_i` |
|---|---|---|---|---|
| B3M | transition | **42/42** | 2.11 | 13.8 pc |
| B3M | momentum | **34/34** | **5.55** | **72.7 pc** |
| B3MW01 | transition | **32/32** | 3.53 | 21.3 pc |
| B3MW01 | momentum | **25/25** | 4.58 | 32.0 pc |

**A 72.7 pc ionisation front in a 5.0 pc cloud — 14.5× the cloud radius**, on a model whose whole
premise is an ionised layer confined between the wind cavity and the neutral shell.

**D3 — and the runs are marching toward the singularity, not away from it.** `drive/P_conf` = 6.333
(B3M) → **15.265** (B3MW01, `Lw`×0.1), ratio **2.410** against the `P_conf^{−1/3}` prediction of
2.154. Extrapolating one more decade to **`B3MW001` — a registered config in the matrix that has
never been run for K10** — gives `drive/P_conf` ≈ 29.4 and `R_i` ≈ **42 pc**, i.e. 8× the cloud.

**Net: the finding is stronger, not weaker.** The old framing ("K10 fails an artificial photo-only
fixture") was dismissible exactly as the maintainer suspected. The measured statement is not: **K10
places its ionisation front outside the neutral shell on 18/18 momentum rows and outside the cloud on
100% of driving rows in both screened configs.** The broken Spitzer test is the *symptom*; the
disease is that the CEM's geometric amplification `(R_i/R2)²` is unbounded and trinity's runs are
already deep in the regime where it produces fronts with no gas to be a front in. Severity
**re-labelled: not CRITICAL-runtime, but MAJOR-domain**, and it is now the same finding as slice 2's
mass over-subscription and slice 1's `R_i > rShell` — three slices converging on one defect from
three directions. Slice 2 already priced the fix: capping the front at the mass that exists takes
`drive/P_conf` from 9.770 to **5.49**, a 1.78× cut.

⚠️ **Consequence for the "re-point the screeners at core-6" recommendation: DOWNGRADED.** I proposed
it when the picture was "K10 looks sound, coverage is thin". The picture is now "K10 is outside its
domain on 100% of driving rows in both configs screened". Running four more configs would measure the
same hole in more places at a cost of hours of wall-clock, and would not change any decision on the
table. **It is no longer the cheapest useful next step.** What is: decide whether the front should be
capped (at the shell edge, at `rCloud`, or by available mass) — a modelling decision that changes K10
fundamentally and whose cost slice 2 has already measured.
#### Slice 1 RESULT — implementation numerics. Baseline is CLEAN; two one-line defects; and my own suspicion was WRONG

**Baseline first, because it matters:** on shipped settings the function is correct. **436/436**
`b17` rows and **3490/3490** real states harvested from the nine committed runs under `outputs/`
solve, worst error **5.0e-13** against 60-digit `mpmath` and **5.0e-13** against Batch 17's
independent `solve_ivp`. The closure's arithmetic is sound where trinity actually runs.

**F1 — MAJOR (latent): my guard tests the wrong quantity, and is therefore unreachable.** I wrote
`if k*(hi−R2) < 1e-8: return hi`, i.e. a test on `τ`. The cancellation is governed by **`x = k·R2`**:
the error behaves as `eps/τ · (1 + 2/x + 2/x²)`, and the guard never sees the `2/x²` blow-up.
Measured by rescaling `dust_sigma` across the 3490-state corpus, **the guard fired 0 times at every
scale from 1× down to 1e-20×** — it only ever engages when `read_param.py:369` sets `dust_sigma = 0`
exactly. The function degrades roughly **five decades before** the guard would trigger. ⚠️ This
supersedes slice 3's suggested remedy ("raise the threshold to ~1e-3"): raising a threshold on the
wrong variable does not fix it.

**F2 — CRITICAL where reachable: `φ(hi) ≥ 0` in float64, so the "guaranteed bracket" is not
guaranteed.** My docstring claims *"the no-dust radius is a GUARANTEED upper bracket … there is no
non-convergence branch"*. True in exact arithmetic, **false in floating point**. At `x ≲ 1e-5` the
computed `φ(hi)` is `+1.0` — the dust term has vanished into rounding — and `brentq` raises
`ValueError`. Worse, at `x = 7.4e-7` it returns `−34` (should be `O(−1e-6)`) and brentq **converges
successfully on a root 8.9% wrong, silently**. Reachability is via metallicity: `dust_sigma =
1.5e-21·ZCloud`, safe at the `dust_noZ = 0.05` default, but lowering `dust_noZ` to keep dust on in a
metal-poor cloud walks in — **14 rows raise at `ZCloud` = 0.005, 102 at 0.001, 164 at 0.0005**.
**Blast radius is asymmetric and bad:** four of the six call sites are unwrapped
(`run_energy_phase.py:227`, `run_momentum_phase.py:636`, `run_energy_implicit_phase.py:983`,
`run_transition_phase.py:566`) so the `ValueError` kills the run — while the other two
(`run_energy_implicit_phase.py:1382`, `run_transition_phase.py:848`) sit inside `except Exception`
blocks that **swallow it** and silently skip the phase-boundary reconciliation snapshot. A crash is
better than that.
**Both F1 and F2 close with one algebraically identical line**, factoring the cancellation out with
`expm1` (`P(s) = (x²−2x+2)/k³` has no internal cancellation, so the entire error is one subtraction):
`bracket = −expm1(−u)·P(R2) + (2·R2·d + d²)/k − 2d/k²`. Zero failures at every scale tested.

**F4 — MINOR: two unguarded exceptions C3c did guard.** `Eb = 1e-300` → `n0**2` **underflows to
exactly 0.0** → `3.0/A` → `ZeroDivisionError`; `Eb = 1e120` → `OverflowError` at `n0**2`. `get_phii_c3c`
guards the analogous divide (`if not (denom > 0.0 …): return 0.0`); K10 does not. Physically
unreachable (89 decades of margin, F5) but one `and A > 0.0` from closed.

**✅ NON-ISSUES, checked hard.** **Negative returns are impossible** — not merely unobserved but
*proven*: the `P_ram` inside `get_effective_bubble_pressure` is the bit-identical expression that
gets subtracted, and IEEE rounding is monotone, so `fl(P_conf·ρ) ≥ P_conf ≥ P_ram`; 7k random states
gave 0 negatives. **Magnitudes are fine** — `n0**2` peaks at 1.79e130 against a 1.798e308 ceiling,
89 decades of margin. **Cost is invisible** — 23.9 µs/call against ≥4.1 ms for a single shell
`odeint` slice, i.e. **0.6%**, with brentq at median 9 / max 15 iterations. (Minor brittleness: my
`rtol=1e-15` sits only 1.13× above scipy's hard floor of `4·eps`.)

**✏️ F7 — MY OWN SUSPICION WAS WRONG, and I am recording that.** In this batch's registration I
flagged the unused `shell_props` + the `n_IF_Str > 0` call-site gate as "a REAL suspected defect".
Measured: **`n_IF_Str == 0` on 0 of 3490 rows across all nine committed runs.** Its only live route is
shell dissolution, which is run-terminating, so exposure is at most the final step. **The gate is
MINOR, not the critical issue I implied** — real dead coupling worth deleting, but not a hazard.
⚠️ **The genuine cost of the discarded argument is different and was not in my docstring:** K10 drops
C3c's `Qi_abs = Qi·shell_fAbsorbedIon` and assumes the full budget in a medium of unbounded extent,
so **`R_i` lands outside the run's own `rShell` on 14–100% of rows depending on config** (100% on
`orionM43_exC` and `log_eyeball`, 56% on `cloud_example_homogeneous`). That independently corroborates
slices 2 and 4 from a third direction.

**⛔ F8 — MAJOR review hygiene: a second broken pin, and a test that now passes VACUOUSLY.** The patch
header discloses only `test_phii_c3c.py`. Also broken: `test_phii_c3c_spitzer.py` (already confirmed
in the CRITICAL finding above). **And `test_mu_audit_drift.py` still passes without testing anything
live** — its assertion matches the now-dead `def get_phii_c3c` body, since the patch is purely
additive and only rebinds the module attribute; the live path uses the reciprocal
`mu_ion_shell/mu_convert` and matches nothing, while `assert calls == 6` counts call sites that no
longer reach the audited function. The factor itself is correct (it is exactly
`shell_structure.py:125`'s `nShell0`), but **the audit no longer audits the code that runs** — a green
test giving false assurance, which is worse than a red one.

**Two things I would not run the arm without**, both one-liners: the `expm1` refactor (kills F1+F2
across five decades of `ZCloud`) and a `try/except (ValueError, RuntimeError): return hi` around the
`brentq` call — because two call sites currently turn the failure into a swallowed warning. And the
docstring's "there is no non-convergence branch" must be struck: it is an exact-arithmetic claim
presented as a floating-point guarantee.

#### Slice 2 RESULT — "all four seams absent by construction" is FALSE for K10 as implemented. ⛔ **ARM-BLOCKING.**

Adversarial audit, then **headline independently re-verified by me** before recording: `mu_convert`
read from the param matches the audit's inferred value to all digits (1.178319e-57), and the two
deciding numbers reproduce exactly — at `t` = 1.5 K10's layer needs **253,412 Msun** against a shell
of **101,805 Msun** (ratio **2.4892**), with the crossing at `t` ≈ 1.089 (1.0603). (My row count was
7/34 vs the audit's 11/34 purely because I joined to `b11_mass_ledger`'s available rows while the
audit used an analytic `shell_mass(R2)` validated to 4.4e-5 over all 34 — coverage, not
disagreement.)

**✅ Seam A — ABSENT, as claimed.** `_k10_front_radius` integrates *literally* `get_shellODE.py:120`:
same equation, same start `r = R2`, same `φ(R2) = 1`, same `σ_d`. `f_abs` appears **zero times** in
the patch. C3a's defect was *spatial* — it spent the **shell's** absorbed budget on a Strömgren
sphere over the **disjoint cavity**, which is why B11.0 measured ≈`2·Qi`. K10 has no cavity term, so
it is a reduced model **of** the shell solve's sink, not a second one. This claim survives.

**⛔ Seam C — PRESENT, WORSE THAN C3c's, and newly present on the branch C3c kept clean.**
`m_implied = (4/3)π(R_i³ − R2³)·n₀·μ_convert`, two routes agreeing to 4e-16:

| config | phase | n | median `m/shell_mass` | max | rows > 1 |
|---|---|---|---|---|---|
| B3M | momentum | 34 | 0.4835 | **2.4892** | **11/34** |
| B3M | **energy** | 87 | **6.3869** | **127.71** | **68/87** |
| B3M | transition | 42 | 0.1019 | 0.1618 | 0/42 |
| B3MW01 | **energy** | 69 | **10.109** | 96.89 | **68/69** |
| B3MW01 | momentum | 25 | 0.3538 | 0.6277 | 0/25 |

Apples-to-apples with C3c's published figure: C3c `(M_cav+M_shell)/M_avail` = **1.5638**; K10
`m_implied`/(all gas inside `R_i`, ISM included) = **1.628**. **The seam is not removed — it is moved
from the cavity into the layer and made larger.** Onset is a size threshold, `R2 ≈ 15.1 pc`
(`m ∝ R2²`), which is why B3MW01's momentum never trips it (max `R2` = 7.73 pc) — regime-scoped
exactly as Batch 12 scoped C3c's seam C. Dust does not rescue it: no-dust 532,680 Msun → dusty
253,412 Msun, still 2.49× the shell. And §6b item 5's "the confined branch is exactly
self-consistent — no cavity gas mass is claimed" **no longer holds**: K10 returns `P_conf·ρ > 0`
there (G18.1) and its implied mass exceeds the shell's on 68/87 and 68/69 energy rows.

**⛔ Two NEW seams, both of the §6b seam-B type.**
1. **Front-radius double-solve.** `n₀` **is** `shell_n0` (ratio 1.0000 exactly on every
   implicit/momentum/transition row), so the two models agree at `r = R2` and then diverge: K10 holds
   `n₀` uniform while the real profile rises outward (`n₀/n_rms` = 0.611 median), and the same `Qi`
   and same ODE yield **two different fronts** — K10 `R_i/R2` = **2.493** vs the shell solve's
   measured `R_IF/R2` = **1.933** (ratio 1.291, up to 1.451). Because the drive is `(R_i/R2)²`, that
   is a **1.667× median (up to 2.104×)** inflation over the same closure evaluated on the run's own
   front. ⚠️ **K10's ionisation front lies outside the shell's entire outer edge on 34/34 B3M
   momentum rows** (`R_i/(R2+dR_full)` median 1.282).
2. **Energy-phase boundary-pressure mismatch, newly introduced.** C3c read the same un-ramped `Pb`
   the shell solve uses at `shell_structure.py:125`, so they agreed exactly. K10 deliberately uses
   the **ramped** pressure (G16.3, to avoid re-admitting D-ramp), so `n₀/shell_n0` = **0.3325–1.0732
   (median 0.8167)** on B3M energy: inside the ramp window the closure and the shell solve now
   disagree by up to **3×** about the density of the same gas at the same radius. Knowingly traded,
   but not "absent".

**🔑 Why this is structurally worse than C3c's seam B, and the finding that matters most.** Under
C3c, `M_cav` was a *diagnostic consequence* — the drive did not depend on the cavity geometry. Under
K10 **the drive IS the geometry**: `P_drive = P_conf·(R_i/R2)²` delivers force `4πR_i²·P_conf`, and
that amplification exists only because the model places an ionised layer of mass `m_implied` out to
`R_i`. So seam C and the drive magnitude are **the same number**. Capping the front at the mass the
run actually has gives `R_i,cap/R2` = 2.343 vs K10's 3.126 at `t` = 1.5, i.e. `drive/P_conf`
9.770 → **5.49**. **K10's drive is 1.779× the largest mass-consistent value, and you cannot fix its
mass book without cutting its drive.**

**Caveat carried from the audit:** all of this is offline, evaluated on a **C3c-driven** trajectory.
The mechanism is an `R2` threshold and G17.4 puts K10's B3M momentum drive at 0.884× shipped, so a
K10-driven B3M would plausibly reach comparable `R2` and trip the same threshold — but that is
inference. G18.2/G18.3's ladder would settle it.

**Disposition: ⛔ ARM-BLOCKING under this batch's registered bar.** The blanket claim "all four §6b
seams absent by construction", as applied to **K10 as implemented in trinity** (as opposed to Geen
et al.'s self-contained two-equation model, where it may well hold), is **withdrawn**. §7.1's K10
row and §Batch 13 both repeat it and are corrected.

#### Slice 0 (run by me while the four were in flight) — the +34% state-jump reproduces, but it is measuring the wrong thing for the continuity argument

The `+34.0%` shipped state-jump is the headline number behind "K10 is smoother", so I re-derived it
from `data/b7_regime_trajectory.csv` independently. **It reproduces to the digit**: switch row
`t = 0.3012` (transition), below `1.3494e3`, above `1.8079e3`, **+33.98%**. Batch 13's arithmetic is
sound.

⚠️ **But the number answers a different question than the continuity claim asks, and Batch 13's own
correction of it looks wrong.** Batch 13 wrote: *"the shipped state-jump measures +34%, not the 23.4%
`P_ram/Pb` figure quoted earlier — that earlier number was the wrong ratio for this quantity."*
Measured at that row, `P_ram/Pb` = **0.2339**, i.e. the discarded figure was **23.4%** exactly. Both
numbers are real; they measure different things:

- **The scheme's intrinsic discontinuity, at the crossing.** As `P_C3a → Pb` from below, the
  transition drive goes `max(Pb, 0 + P_ram) = Pb` → `max(Pb, Pb + P_ram) = Pb + P_ram`. The jump is
  therefore **exactly `P_ram/Pb` = +23.4%**, analytically, with no sampling involved.
- **The +34%** is that same jump evaluated at the first *sampled* row after the crossing, where
  `P_C3a` has already overshot `Pb` by a factor **1.106**. So it is the intrinsic discontinuity
  **plus one sampling interval of real evolution** — which is the *same* defect that got G13.1's
  original metric thrown out ("adjacent snapshots are separated by real evolution, so that metric
  conflates the discontinuity with genuine change"). Evaluating both branches at one fixed state
  removed only half of that problem: the state chosen is still post-crossing.

**Disposition: record-correcting, not arm-blocking.** K10's jump is 0 by construction either way, so
the qualitative claim ("K10 removes a discontinuity the shipped scheme has") stands. What changes is
its **size**: the discontinuity K10 removes is **+23.4%** at the crossing, not +34%, and the +34%
figure should be quoted as "the branch's effect at the first post-switch snapshot" rather than as the
scheme's discontinuity. Batch 13's dismissal of 23.4% as "the wrong ratio" is withdrawn — it was the
right ratio for the continuity question and the wrong one for the question Batch 13 was then asking.


---

### Batch 21 — K10-O1: use the shell solve's own ionisation front — Status: ✅ **screened 2026-08-29 — G21.0/G21.1/G21.2 PASS; G21.3's disclosed prior MISSED by 16%; G21.4 the confined excess survives at half; G21.5 confirms what O1 does NOT fix**

**Maintainer decision 2026-08-29**, after Batch 20 priced the four remedies for K10's unbounded
front: **take O1.** Stop solving a second front; read the one trinity already computes.

**The change.** `rho = (shell_props.R_IF / R2)**2`, and **`_k10_front_radius` is deleted**.
`R_IF` is a first-class field on `ShellProperties` (`shell_structure.py:39`, returned at `:471`,
set at `:227` as `rShell_arr_ion[-1]`), so this is a read, not a re-derivation. Everything else —
`P_conf` from the ramped `get_effective_bubble_pressure`, and Batch 16's composition mapping — is
unchanged.

**Why this is more than a cap.** Deleting the front solve removes, *by construction rather than by
patching*: slice 1's **F1** (the guard tests `k·(hi−R2)` when cancellation is governed by `k·R2`, so
it never fires), **F2** (`φ(hi) ≥ 0` in float64 → unhandled `ValueError`, or a silently 8.9%-wrong
root), **F4** (`ZeroDivisionError`/`OverflowError` on extreme `Eb`), slice 2's **new seam 1** (two
models solving two different fronts from the same `Qi` and the same ODE), and Batch 20's **domain
violation** (`R_i` beyond the shell on 18/18 momentum rows, beyond `rCloud` on 100% of driving rows).
It also makes **`shell_props` load-bearing again**, which retires slice 1's F7 dead-coupling finding
and gives the vestigial `n_IF_Str > 0` call-site gate something real to gate (**B11.E**).
⚠️ **And it makes Batch 17 moot as machinery**: the shell solve's `R_IF` already carries dust
(`get_shellODE.py:120`), so the closure's own dust model is no longer needed. Batch 17's value is
retroactively **validation** — it showed the closure's dust matched the shell solve's to 5.6%, which
is now an argument that the shell solve's front is the right thing to read, not a component we ship.

**Gates.**
- **G21.0 — it IS the code's own front (BLOCKING).** The value used must be `shell_props.R_IF`
  verbatim, with no rescaling, clipping or re-derivation, and `_k10_front_radius` must be gone.
  *Falsifier:* any transformation applied to `R_IF` before use.
- **G21.1 — the domain violation is gone (BLOCKING).** `R_i ≤ R2 + shell_thickness` on **100%** of
  rows. Batch 20's D1 measured the current form violating this on 18/18 B3M momentum and 43/44
  energy rows; O1 must take that to 0/0. *Falsifier:* any row beyond the shell.
- **G21.2 — seam C.** The implied ionised-layer mass must now be a *sub-part of the shell's own
  mass*, so `m_implied/shell_mass ≤ 1` on every row. Slice 2 measured the current form at median
  0.4835 / max **2.4892** in B3M momentum and median **6.39** in energy. *Falsifier:* any row > 1 ⇒
  the mass argument is still broken and O1 has not fixed what it was chosen to fix.
- **G21.3 — magnitude (measurement, no bar).** `drive/P_conf` and composed/shipped per phase per
  config. Disclosed prior from Batch 20's pricing: B3M momentum **3.901** (`R_IF/R2` 1.975), i.e. a
  ×1.62 cut from K10's 6.333 and ×1.82 below shipped C3c's 7.095.
- **G21.4 — what happens to the confined branch.** In the ED phases the shell is thin, so
  `R_IF/R2 → 1` and the excess → ~0. **Pre-registered expectation: O1 largely restores C3c's
  confined behaviour**, which would mean Batch 16's "Lancaster's first-order term is now delivered
  at +0.96%" argument **evaporates under O1**. Report the ED excess; no bar, but if it is ≈0 then
  the case for K10 over C3c on the confined branch is gone and must be re-stated honestly.
- **G21.5 — what O1 does NOT fix, stated so nobody reads it as a general repair.** (a) The
  **photo-only limit is still broken** — the drive is still ∝ `P_conf`, so `P_conf = 0` still
  returns 0.0 and `test_phii_c3c_spitzer.py` still fails. (b) The **per-segment freeze ratchet** is
  untouched and independent. (c) Coverage is still two configs of one cloud. Report all three.

**Out of scope:** the freeze ratchet (needs a live-call decision), coverage, default flip, D5.

---

#### Batch 21 RESULT — 2026-08-29, measured against the gates above

Reproduce: `python docs/dev/phii-identity/harness/k10_o1_screen.py`. Arm patch:
`hpc/b14/k10_o1_arm.patch` (applies clean; supersedes `k10_arm.patch`). 189 rows.

- **G21.0 ✅ PASS.** `R_IF` is read verbatim from `ShellProperties.R_IF` with no rescaling or
  clipping, and `_k10_front_radius` is **deleted** — verified in the built module: the attribute is
  gone and no `brentq`/`scipy.optimize` call survives in the code path (the word appears only in the
  docstring). Slice 1's F1/F2/F4 are therefore closed **by deletion**, not by patching.
- **G21.1 ✅ PASS — the domain violation is gone.** Front inside the shell on **140/140** rows
  (B3M energy 0/44, implicit 0/34, transition 0/41, momentum 0/21 beyond). The Batch 18 form
  violated this on **18/18** momentum and **43/44** energy rows.
- **G21.2 ✅ PASS — seam C is closed.** Implied layer mass ≤ shell mass on **every row screened**;
  worst case 0.9407 (B3M energy), B3M momentum median 0.1261 / max 0.7724. ⚠️ **Coverage corrected
  2026-08-29:** an earlier wording here said "every phase and both configs". It is **B3M-only in the
  ED phases** — `b21_o1_screen.csv` carries no B3MW01 energy/implicit rows, because the low-wind
  photon ledger replays driving rows only. The same caveat applies to G21.4's ED median.
  The Batch 18 form gave momentum median 0.4835 / **max 2.4892** and energy median **6.39**.
- **⚠️ G21.3 — measured, and my disclosed prior MISSED.** I pre-registered "B3M momentum 3.901" from
  Batch 20's pricing; measured **3.274**, a 16% miss. Recorded as a miss. Cause is not an error but a
  row-set difference: Batch 20 took the median over `b9`'s own 17 momentum rows, while this screen
  joins `b17`'s 21 momentum rows to the nearest `b9` front, and the two span different `t`.

| config | phase | O1 `ρ` | K10 `ρ` | O1/K10 | **O1/shipped** |
|---|---|---|---|---|---|
| B3M | energy | 1.001 | 1.006 | 0.995 | **1.001** |
| B3M | implicit | 1.005 | 1.005 | 1.000 | **1.005** |
| B3M | transition | 2.228 | 3.420 | 0.645 | **0.462** |
| B3M | momentum | 3.274 | 5.415 | 0.600 | **0.494** |
| B3MW01 | transition | 3.947 | 10.915 | 0.363 | **0.340** |
| B3MW01 | momentum | 4.782 | 15.075 | 0.318 | **0.325** |

  **Two things stand out.** The **confined branch is left essentially alone** (O1/shipped 1.001 and
  1.005), so unlike the Batch 18 form O1 does not disturb the branch §6b found exactly
  self-consistent. And the **driving branch is roughly halved against shipped C3c** — ×0.49 in B3M
  momentum, ×0.33 in B3MW01. That is a large, measurable change and it is the thing an arm run
  would show.
- **G21.4 — the confined excess SURVIVES, at about half.** ED median **+0.4751%** over `P_conf`
  (max 0.6130%), against the Batch 18 form's +0.96% no-dust / +0.67% dusty. My pre-registered
  expectation was that it might collapse to ≈0 and take Batch 16's "Lancaster's first-order term is
  delivered" argument with it. **It does not** — the term is smaller but real, so that argument
  stands in reduced form.
- **G21.5 — what O1 does NOT fix, verified under the patch rather than asserted.**
  (a) `test_phii_c3c_spitzer.py` is **still 5 failed / 1 passed** — the drive remains ∝ `P_conf`, so
  there is still **no photoionisation-only limit**. O1 fixes the *geometry*, not the *singularity*.
  (b) `test_phii_c3c.py` **6 failed / 5 passed** — C3c's exact-0.0 confined contract is replaced by
  the +0.475% excess. Broken by design, recorded, not re-baselined. (c) The **per-segment freeze
  ratchet** is untouched and independent.

**Where this leaves K10.** Of Batch 20's five blocking findings, O1 closes **three** outright
(domain violation, seam C, and the numerics defects F1/F2/F4 by deletion) and **retires slice 1's F7**
by making `shell_props` load-bearing again — which also gives the vestigial `n_IF_Str > 0` call-site
gate something real to gate (**B11.E**). It closes **none** of: the missing photo-only limit, the
freeze ratchet, or coverage. ⚠️ And it does **not** resolve D5's magnitude question — at 3.274×`P_ram`
the momentum phase is still photoionisation-dominated, just by ~3× instead of ~7×.

🔑 **A simplification worth stating: Batch 17 is now moot as machinery.** The shell solve's `R_IF`
already carries dust, so the closure's own dust model is not shipped under O1. Batch 17's value is
retroactively **validation** — it showed the closure's dust matched the shell solve's to 5.6%, which
is now an argument for reading `R_IF` rather than a component of the code.

---

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
  ⛔ **[B11.A 2026-08-18 — the CONSEQUENCE clause above is REFUTED.]** "A photon-conserving
  cavity+shell accounting has less than `Qi` available to the cavity ⇒ `P_C3a` overstated (√ of the
  budget)" predicts `P_C3a_fixedpoint/P_C3a_shipped` < 1. Measured: **1.0000–1.1778, 0 of 33 driving
  rows below 1** (G11.A2). Conserving photons *raises* `P_C3a` by up to 17.8%. The double-spend
  stands; "⇒ overstated" is struck, and seam A leaves the upper-bound list. The deeper result is
  that the fixed point is **degenerate** — `x = f_abs(Qi(1−x))` has the unique root `x = 1` on every
  driving row, i.e. the cavity takes every photon and the shell is left neutral (G11.A1), so C3a
  cannot be closed photon-conservingly at all without a second equation.
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
  ⚠️ **[B12 2026-08-18 — REGIME-SCOPED. 0.564 is a B3M number, not a universal one.]** The low-wind
  rung `B3MW01` (`Lw × 0.1`) gives `M_cav/M_shell` = **0.1296** at t=1.5, tripping the falsifier
  registered for it (< 0.2). `M_cav ∝ R2^{3/2}·sqrt(Qi f_abs)`, so the controlling variable is
  **bubble size, not the degree of HII dominance** — that run only reaches `R2` = 7.733 pc against
  23.253 pc. The seam is worst in the configs that expand furthest. **The qualitative core survives
  unchanged**: `(M_cav + M_shell)/M_avail` = 1.1296 (still over-subscribed), wind mass 31.6 Msun =
  0.24% of `M_cav`, and the dynamical cost falls to +0.45%/+0.97%. Always quote this seam with its
  config. `data/b12_lowwind_mass_ledger.csv`, `data/b12_lowwind_mass_dynamics.csv`.
- **D. Thin-shell strain.** `dR/R2` = 0.67–1.31 in momentum (Batch 9) — the ODE's thin-shell
  premise and C3a's sharp cavity/shell split are both at their validity edge there.
  **[B11.0 2026-08-18 — CONFIRMED.]** `dR_full/R2` = 0.6723–1.3078, `dR_ion/R2` = 0.6579–1.3076;
  `dR_ion/dR_full` median 0.9954, so the seam does not depend on which thickness is meant.
  **[B12 2026-08-18 — WORSE at low wind.]** `B3MW01` momentum `dR_ion/R2` = **1.171–1.438**
  (median 1.213). The thin-shell premise is violated harder when the wind is weak, not less.
- Also vestigial, cosmetic + edge-behaviour only: the C3c call is gated by the OLD capped quantity
  (`n_IF_Str > 0`) in every phase runner, and the `F_HII` docstrings still say "from n_IF_Str".

**Direction and consequence.** ~~Every seam pushes the same way:~~ **[B11.0 2026-08-18 — CORRECTED.
Three of the four push the same way, not all four.]** **[Updated again after B11.A–D, 2026-08-18 — the list is down to one seam plus dust.]**
**The shipped driving-branch `P_C3a` is an upper bound** because of **C** (the filled-cavity limb
needs 56% more gas than the run has, so the drive must be the supply-limited limb) **and G9.4's
dust** (−51–75% of the budget). **Neither A nor B belongs on this list.** A's double-spend is real
but repairing it *raises* `P_C3a` by 0–17.8% (G11.A2), not lowers it; B's feedback is exactly zero
on 88% of driving rows and upward on the rest, so it inflates nothing either — it is an
inconsistency in thickness, dust column and gravity sampling. Both original direction claims were
measured and struck. None of this is
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
| D5 | ⬜ **OPEN, and now the load-bearing one (raised 2026-08-16, Batch 9 scope).** **What pressure does a photoevaporative ionised layer transmit to the neutral shell?** C3c drives at the full `n_tot k T` of the ionised gas. `c43a50e`'s own commit message flags this as unexplored: *"a photoevaporative flow does not drive at n k T of the whole region."* Batch 8 removed the calibration explanation (C3a's magnitude **is** the classical D-type pressure, exactly). Batch 9 removed the geometry explanation too, but **not** for the reason its scope first claimed: the correction is *not* one-signed — in the momentum phase the shell is thick (`dR/R2` 0.67–1.31) so the layer form **lowers** `P_HII` 0.51–0.71×. It just is not enough: `P_HII/P_ram` median 6.165 → 3.594 on the analytic layer form, and **→ 1.545 (1.322–1.666, falling with time)** on the profile form that G9.4 shows is the trustworthy one — still HII-dominated on every row, but by ~50% rather than ~500%. So geometry moves the right way and lands close to unity without crossing it, and this is what is left. ✅ **Batch 10 closed the one escape route that remained**: the `Lw ≈ 3.4` inversion extrapolated from stage 3's exponent was tested on `B3MW3`/`B3MW10` and **falsified** — the profile form falls only as `Lw^−0.1133` because stronger winds thin the shell and that raises the geometry correction, so inversion sits at `Lw` ≈ 46.5, still unphysical. **D5 is therefore load-bearing on measurement, not just on absence of alternatives.** Candidate answers, none measured: (i) full `n k T` — the status quo; (ii) a momentum-flux-limited transmission (the flow carries `rho v^2`, not `n k T`); (iii) C3's never-implemented option (c), `P_ram + max(P_C3a − P_conf, 0)` — transmit the confining pressure, add only the excess. **This is a physics-intent call, not a code call**, and it cannot be settled by measurement.<br><br>**Option (iii) analysed 2026-08-17 (maintainer question).** In the momentum phase `P_conf` **is** `P_ram` (`run_momentum_phase.py` assigns `Pb` = ram pressure), so option (iii) collapses to an identity — verified to machine precision on all 34 B3M momentum rows:<br>`P_ram + max(P_C3a − P_ram, 0)` ≡ `max(P_C3a, P_ram)`<br>i.e. **in momentum, option (iii) IS C1 ("transmit, don't add") applied to C3a.** C1 was measured in Batch 3 and judged "safe, but aimed at the wrong target" — but only because `P_HII ≡ P_ram` then made `max(P_HII, P_ram) = P_ram`, which *deleted* the photoionised channel. With C3c's decoupled `P_C3a` it deletes nothing, so **Batch 3's objection to C1 no longer applies**. Outside momentum they differ: transition's `P_conf` is the bubble thermal pressure, so `P_ram + max(P_C3a − Pb, 0)` is not a plain max there.<br><br>**Measured effect (B3M momentum, 34 rows):** shipped `P_drive/P_ram` median **7.095** (6.083–8.161) → option (iii) **6.095** (5.083–7.161) = a **14.1% reduction**, exactly the `+P_ram` double-count and nothing else. ⚠️ **So option (iii) fixes additivity, not dominance** — the drive is still ~6× ram because `P_C3a` dominates alone. Of the three options, **(ii) momentum-flux-limited is the only one with factor-level leverage on magnitude** (a photoevaporative flow carries `rho v^2`, not static `n k T`).<br><br>**Tension with D1, which must be settled to adopt (iii).** D1 ruled the momentum sum *intended*, conditional on `P_HII` being genuinely its own calculation — a condition C3c now satisfies, so by D1's own logic the sum is legitimate and (iii) reverses it. The counter is physical rather than bookkeeping: if the ionised skin is what contacts the shell, the wind pushes the *skin* and the shell feels the *skin*, so adding both double-counts the wind regardless of decoupling. Genuine disagreement between the two framings; needs the maintainer.<br><br>**Maintainer direction 2026-08-17:** "transmit, don't add" is favoured, with the `P_ram + P_HII` form retained as a **selectable key** and the default *possibly* moving to transmit. **No code written — two design decisions are deliberately parked with the maintainer:**<br>**(a) the key's default.** `'sum'` first is provably byte-identical and needs no re-verification, leaving the flip as its own gated change; defaulting straight to `'transmit'` is a **default flip**, which CLAUDE.md's planning protocol classes as risky/outward-facing — pre-registered gate, full-run equivalence on the stiff regimes in separate processes at matched `t`, plus D4 golden re-baselines.<br>**(b) the scope.** Momentum-only is 2 sites (`run_momentum_phase.py:265,445`), matches Batch 3's measured C1 arm, and respects D1's ruling that the transition `max` is a deliberate `Pb → 0` handover. Momentum+transition is 5 sites (adding `run_transition_phase.py:331`, `energy_phase_ODEs.py:253,385`), becoming `max(Pb, P_HII, P_ram)` **[CORRECTED 2026-08-18, from rev2 of the external assessment, verified against source here: the site COUNT above conflates live and reporting sites. `run_transition_phase.py:331` and `run_momentum_phase.py:265` are inside `compute_forces_pure`/`compute_forces_momentum_pure` — reporting only; the LIVE drive sites are `energy_phase_ODEs.py:253` (1c, via `get_ODE_Edot_pure`, delegation verified at `run_transition_phase.py:231`) and `run_momentum_phase.py:445` (the `get_ODE_momentum_pure` RHS). Rev2 further claims `:385` is unreachable — not independently verified here. Any future scope decision counts live sites: momentum-only = 1, momentum+transition = 2.]** — the same double-count is arguably the same defect there, but it has no prior measurement and touches a construct D1 called deliberate. The energy/implicit sites (`energy_phase_ODEs.py:388`, `run_energy_implicit_phase.py:532`) are `max(Pb, P_HII)` with no `P_ram` and are **unaffected either way**.<br>⚠️ Whichever lands, this is the case where a `.param` key **is** warranted — unlike the capped-Strömgren prescription (see §2a discussion), `sum` and `transmit` are two defensible physics choices, directly analogous to `betadelta_solver`. <br><br>**Superseded 2026-08-18 — transmit is DROPPED; the candidate is now 'C3a-raw' (maintainer direction).** Two follow-up questions from the maintainer: (1) is the confined branch's exact-0.0 right, or should the helper always return `P_C3a` and let the existing `max`/sum structure decide dominance ("if it's smaller than Pb it should just have subdominant contribution")? (2) momentum should stay *independent Strömgren pressure + `P_ram`* — **which it already is**: the switch is inert in momentum on every measured config (`P_C3a > P_ram` on 100% of momentum rows), so momentum needs no change under either scheme. Measured effect of C3a-raw (always-return) vs shipped C3c, on committed runs — reconstruction validated to 2.2e-16 on B3M's 66 driving rows; ±5% on B3MW001's (value-lag at the rapidly-evolving early steps):<br>• **B3M (nominal wind): drive-identical in energy (0/87 rows), implicit (0/68) and momentum (0/34)**; 1/42 transition rows changes (+7.9%) — and it **removes the `t_cross` kink registered in §3c.1**: stored `P_drive` jumps +6.8% across the switch row, C3a-raw is continuous (−1.0%).<br>• **B3MW001 (weak wind): 11/51 energy rows change, ×2.12 median, ×2.62 max — every one inside the `dt_switchon` ramp window (t ≤ 6.8e-5 Myr)**, where `P_C3a` is below the un-ramped `Pb` (0.49–0.94, genuinely confined) but above the *ramped* `press_bubble`. Always-return therefore re-admits the D-ramp defect class C3c fixed as a side effect — a pressure carrying the un-ramped scale winning the `max` inside the ramp window. Batch 4a showed early-window drive changes can retain double-digit ΔR2 downstream (F1LO 14.4%), so this needs the offline screen + a full-run gate on the weak-wind configs before landing.<br>Design consequences if C3a-raw is adopted: (a) the **D1 tension dissolves** — the momentum sum stays, nothing is reversed; (b) the **scope question dissolves** — C3a-raw is one change in `get_phii_c3c` (return `P_C3a` unconditionally), zero `P_drive` edits, each phase sorts itself via its existing `max`/sum; (c) the pinned exact-0.0 contract (`test_phii_c3c.py`) and Batch 5's pre-registered null (`P_HII`=0 on implicit rows) are deliberately superseded — D4-style before/after bookkeeping required. Remaining decisions: the key's default (`c3c` switch vs `c3a` raw) and how to handle the weak-wind ramp window. | the momentum-phase verdict; any further C3 work | **open — see §7.1 for the candidate register (the single live list); transmit dropped 2026-08-18; K2 (C3a-raw) is the maintainer's candidate; default + ramp-window handling parked** |
| D4 | ✅ **ANSWERED 2026-08-13** — re-baselining authority **granted** for `test_phase_boundary.py`, `test_betadelta_hybr_stress.py` and `test_scheme_screen.py` fixtures, conditional on G3.4: every re-baseline lands with a committed before/after table and the mechanism named. A golden that moves for an *unexplained* reason is still a stop, not a re-baseline. | Batch 6 | **answered** |

### §7.1 D5 candidate register — added 2026-08-18 (reconcile pass)

**Why this exists.** D5's cell above grew by accretion across ~six sessions, and Batches 9–12 plus
the external assessment added four more candidates without anyone writing them in one place. This
table is now the **single list of live options**; D5's prose is the history of how each arrived.
Any new candidate gets a row here or it does not exist. Nothing below is a decision — D5 is still
the maintainer's call.

**Evidence tiers**, so the register cannot be read as if every row were equally supported:
**M** = measured in this workstream · **S** = supported by primary sources (papers read here) ·
**A** = from `LITERATURE_ASSESSMENT.md` (rev2 landed 2026-08-18; its do-not-act banner is lifted, but the tier still means *not independently measured here*) and never load-bearing
(§0 C-0.5) · **U** = unmeasured idea.

| id | candidate | what it changes | tier | status |
|---|---|---|---|---|
| **K1** | **C3c switch** — return `0.0` while `P_C3a ≤ P_conf`, `P_C3a` above | nothing; this is production since `c43a50e` | **M** | **SHIPPED.** Confined branch exactly self-consistent (§6b); driving branch carries the four seams |
| **K2** | **C3a-raw** — always return `P_C3a`, let each phase's existing `max`/sum decide | one return statement in `get_phii_c3c`; zero `P_drive` edits | **M** | **Maintainer's current candidate.** Drive-identical to K1 on B3M energy/implicit/momentum; 1/42 transition rows (+7.9%) and it removes the §3c.1 `t_cross` kink. ⚠️ Re-admits the D-ramp class on weak wind (11/51 `B3MW001` energy rows, ×2.12 median, all inside the `dt_switchon` window) |
| ~~K3~~ | ~~**transmit** — `P_ram + max(P_C3a − P_conf, 0)`~~ | ~~2–5 call sites~~ | **M** | ⛔ **DROPPED 2026-08-18** (maintainer). Collapses to `max(P_C3a, P_ram)` in momentum, i.e. C1 on C3a; fixes additivity (−14.1%) but not dominance |
| **K4** | **Momentum-flux-limited** — the photoevaporative flow transmits `ρv²`, not static `n k T` | `get_phii_c3c`'s value, all phases | **U** | **Unmeasured, and the only candidate with factor-level leverage on magnitude.** Named in `c43a50e`'s own commit message as the unexplored question. No design, no gate |
| **K5** | **Layer geometry** — balance recombination over the ionised layer `(R_i³ − R2³)`, not the cavity `R2³` | the denominator in `get_phii_c3c` | **M + S** | **Strongest external support of any row.** Four independent sources use the cavity-subtracted volume — Lancaster `eq:ionreceq2`, Geen 2019 `wind:photoequilibrium`, Geen 2022 `eqn:photoionisation_equilibrium_uniform`, **and trinity's own `shell_structure.py:243`** — so K1/K2 are internally inconsistent with `n_IF_Str`. Measured effect: momentum `P_HII/P_ram` 6.165 → 3.594 (analytic layer) → **1.545** (profile form, the trustworthy one per G9.4). Does **not** reach unity; ⚠️ the analytic-form numbers carry B11.0's S1 thin-shell bias and B11.F re-fits them. **Rev2 (2026-08-18) ranks this #1 of its "ranked by measured impact" list** — "K5 as the minimal move, K6 as the right one". **Gated as Batch 14 (2026-08-27)**, two variants (K5a analytic layer / K5b profile), with a **blocking decoupling pre-gate** — K5b reads a profile whose inner boundary is `nShell0 ∝ Pb`, so re-introducing the original `P_HII ≡ P_conf` circularity is a live risk. ⚠️ **Screened offline 2026-08-28 (Batch 14 result): the risk is measured real for BOTH variants** — on driving rows the candidates are ∝`Pb^1.0` (log-log r² 0.988/0.993) at gain ≈2.4/1.5 (K5a **is** the uncapped `n_IF_Str_raw`, whose coupling Batch 3b measured at r ≥ 0.997 on 788 rows), and a bare swap flips **83/83 confined rows to driving** (K5b there = 1.07×`Pb` — the old identity to 7%), re-admitting the D-ramp class at Batch-4a-measured cost. The layer volume belongs to the coupled closure (K10/K6). ⛔ **PARTLY WITHDRAWN 2026-08-29:** this row's rejection rested on two grounds and **ground (a), `Pb`-slaving, no longer stands** — the D5 reframe established that pressure-slaving is *physics* when structural (the layer's density IS set by the `nShell0 ∝ Pb` boundary condition, quasi-hydrostatic in Rahner+2017 and assumed explicitly in Lancaster+2025), so K5b's `Pb`-dependence is that statement, not a relabelling. Ground (b) — the additive composition giving ≈2.5×`P_ram` — survives, but it is **D5's additivity question, not a verdict against K5**. Batch 19's check further showed **K5b IS C3a corrected for volume and dust** (`n_rms/n_cav = f_dust·f_volume` identically), giving B3M momentum **1.545×`P_ram`**. **K5b is LIVE again**, not closed |
| **K6** | **Coupled closure** (Geen/Lancaster CEM) — solve one `n_i` and one `r_i` from recombination equilibrium **plus** wind/photoionised pressure balance at `r_w` | replaces the `max`/sum composition in all four phases with a scalar root-find | **S + M** | **Its two central identities are now independently verified here** (`harness/cem_closure_check.py`, 2026-08-18): the C3c momentum switch point **is** `R_ch` to 4.3e-15 over 200 random draws, and the shipped branches are the CEM's exact asymptotes with the crossover error reproducing Lancaster's table to the digit (`F_sum/F_CEM` = 1.3421, `F_max/F_CEM` = 0.6710 at `R_i = R_ch`). **Registered as B11.G, not run on trajectories.** All four §6b seams are absent from it by construction **in Geen et al.'s own formulation** — one photon budget, no cavity mass, and the pressure balance *is* the boundary condition. ⛔ **This does NOT carry over to K10 as implemented in trinity: Batch 20 slice 2 (2026-08-29) measured seam C PRESENT and worse than C3c's (implied layer mass 2.49× the shell), plus two new seam-B instances.** B11.A's degeneracy result (`x* = 1` on 33/33 rows) is the positive argument for it: C3a cannot be closed photon-conservingly *without* a second equation, which is exactly what this supplies |
| ~~**K7**~~ | ~~**`alpha_p` on `P_ram`** — a wind momentum-enhancement factor~~ | — | **M** | ⛔ **DEAD 2026-08-27 — withdrawn by its own author in rev2 ("wrong and is withdrawn in full") and closed by measurement here.** Rev2's four kill-arguments include Batch 12's bound (inversion needs `α_p ≳ 14`, twice Paper II's largest). The decisive measurement: from committed `data/b7_regime_trajectory.csv`, TRINITY's own `(R2/R1)²` — which **is** its α_p-equivalent, per the convention identity verified 2026-08-18 — falls smoothly 44.7 → **1.0018** across the transition phase, landing at 1 to 0.2% exactly when `Eb` hits `ENERGY_FLOOR`. So `α_p = 1` in momentum is what the dynamics *delivers*, not an omission. What survives is a **diagnostic**: emit `(R2/R1)²` (flagged invalid inside `dt_switchon`) plus `ζ`, `R2/R_ch`, `C_w` — rev2 §Diagnostics; registered as part of B11.G. The earlier "straddling parity at α_p = 6.2" arithmetic is superseded: it treated a delivered quantity as a free knob |
| **K8** | **Three-radius model** — `R1 < R_w ≤ R2 ≡ R_i`, with `R_w` algebraic | structural; a follow-up paper | **S** | ⬆️ **Raised A → S 2026-08-28** on a primary-source read: Lancaster+2025 (arXiv:2505.22730) applies the wind at `R_w` and the photoionised gas at `R_i` and calls the distinction central, so this is the source's own structure rather than a speculative row. **Deferred by the maintainer 2026-08-29** ("keep one radius") — a follow-up paper, and the reason O1 borrows Lancaster's force amplification without his inertia placement. Pending. K5/K6 are the minimal ways to get `R_w ≠ R2` without this |
| **K9** | **Shell-mass adjustment** — `M_sh = (4π/3)R_i³(ρ̄ − ρ_i)` | the momentum equation's inertia | **M + S** | **Measured here, and the literature already does it** (Lancaster `eq:pr_spitzer_adj`, with its own "not consistent with the derivation… \[but\] can be more accurate" caveat). B11.C2: **+8.55%/+9.22%** in `R2` at nominal wind, **+0.45%/+0.97%** at `Lw × 0.1`. This is §6b seam C's fix, not a separate idea. **Rev2 ranks it #2.** ⚠️ **Downgraded 2026-08-27 by Batch 15's pre-gate**: B11.C2's +8.55/+9.22% debited **`M_cav`**, which is NOT the K9 quantity — in trinity's geometry K9 debits the shell's own **ionised layer**, measured at `m_ion/m_shell` median **0.461** in momentum (range 0.109–**1.000**), 1.0000 throughout energy. It **reaches a massless shell on real rows**, its admissible phase scope is narrow, and G15.3 may make it inseparable from K5. **Not the cheap independent win it appeared to be** |

| **K10** | ✅ **screened (Batch 13)** — **CEM-interpolated `P_HII`** — the smooth coupled form, phase-agnostic: `n_H0 = (μ_i/μ_c)·P_conf/(k_B T)` (pressure-equilibrium skin density — `shell_structure.py:125`'s own line), `R_i³ = R2³ + 3·Qi_abs/(4π χ_e α_B n_H0²)` (recombination over the cavity-**excluded** layer volume), `P_drive = P_conf·(R_i/R2)²`; equivalently the helper returns the **excess** `P_HII_eff = P_conf·[(R_i/R2)² − 1]` and the existing momentum sum composes it exactly. In the MD phase this is algebraically Lancaster's `α_p ṗ (1 + R_w/R_ch)^{2/3}` at `α_p = 1` | one helper; ⛔ **"zero `P_drive` edits" CORRECTED 2026-08-28 — true in momentum ONLY** (see the composition note in the status cell) | **S + M** | **Registered 2026-08-18, Batch 13 screens it offline.** The momentum-phase minimal form of K6, containing K5's volume fix by construction. Exact limits: confined → excess `= (2/3)(R2/R_ch)·P_conf` (Lancaster's own first-order term — the correct "better than 0.0"); unconfined → Spitzer over the layer volume. **Smooth**: kills the factor-2 momentum switch jump, the 23.4% transition jump, and the §3c.1 `t_cross` kink by construction. ⚠️ Known gaps, pre-stated: **no dust** in the closure (illustrative: predicts `R_i/R2` ≈ 3.1–3.9 on B3M momentum where the shell solve measures `R_IF/R2` ≈ 1.7–2.3 — G13.4 sizes it); assumes quasi-static balance (Lancaster *imposes* it: "we now imagine"); ED-phase use maps `P_conf` to the thermal `Pb` (their §ed_jfb structure) and must respect the D-ramp window. **Batch 13 measured it (2026-08-27):** state-jump **exactly 0** where the shipped rule jumps **+34%**; healthy branch untouched on B3M (+0.68%); but the **dust rule fired at 2.05×**, so **K10 cannot ship without a dust model** — and with dust it lands **within 10–15% of the shipped drive** — ⛔ the "cavity-volume and missing-dust errors partly cancel" gloss on that agreement is **RETRACTED 2026-08-29**: in momentum both corrections deflate on 17/17 rows and compound to ×0.2506, and corrected-C3a (1.545) is 4.10× below K10 (6.333), so K10 is not a corrected C3a at all (see §Batch 13 RESULT). Note `R_ch`(trinity) = `chi_e`·`R_ch`(Lancaster). ⛔ **COMPOSITION CORRECTION 2026-08-28 (algebra, not yet gated).** "The excess rides the existing compositions" holds **in momentum only**, and Batch 13 could not have caught it because its screen computed the total drive `P_conf·(R_i/R2)²` directly instead of routing a helper return through each phase's `P_drive` expression. With `ρ ≡ (R_i/R2)² ≥ 1`: momentum (`P_HII + P_ram`, `P_conf = P_ram`) composes the **excess** exactly (`P_ram(ρ−1) + P_ram = P_ram·ρ`) but the full value over-counts by `P_ram`; energy/implicit (`max(Pb, P_HII)`) is the **opposite** — the **full value** composes exactly (`max(Pb, Pb·ρ) = Pb·ρ`) while the excess is **swallowed by the `max` whenever `ρ < 2`**; transition (`max(Pb, P_HII + P_ram)`) is exact under neither. ⚠️ **This destroys K10's headline confined-branch virtue as specified**: the "correct better-than-0.0" first-order term `(2/3)(R2/R_ch)·P_conf` is *small* by construction, so under `max(Pb, excess)` it is discarded exactly where it was meant to improve on the 0.0. K10 therefore needs a phase-aware helper **or** real edits at the live `P_drive` sites (`energy_phase_ODEs.py:253,256`, `run_momentum_phase.py:445`) — it is **not** a one-helper zero-edit change, and its risk class rises accordingly. Needs its own gate before any K10 arm |

**How the rows relate, so they are not treated as nine independent choices.**
K1/K2/K3 are the same quantity with different *branch logic* — a key, not physics. K4/K5 change the
*value*. K6 replaces the whole composition and **subsumes K5** (its `r_i³ − r_w³` is K5's volume).
K7/K8/K9 are about the wind and the shell, not `P_HII`, and are orthogonal to all of the above —
K9 in particular is already measured and independent of whichever of K1–K6 wins.

**What the workstream's own evidence currently favours, stated as evidence and not as a decision.**
B11.A's degeneracy result is the sharpest input: `x = f_abs(Qi(1−x))` has the unique root `x = 1` on
every driving row, so seams A and B **cannot be repaired inside C3a's structure** at all. That is an
argument about K1/K2 *as a family*, and it points at **K6**. Against that: K6 is the largest change
on the table, it has no gate and no measurement here yet, and B11.G is the cheap way to find out
before committing. ⛔ **Superseded 2026-08-29:** the corrected assessment (rev2) landed 2026-08-18 and its
do-not-act banner is lifted; Batches 13–21 were built after it. What remains open is the
maintainer's ruling on the register, i.e. D5.

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
| 6 | 🟡 | 2026-08-14 | **C3c LANDED (`c43a50e`, PR #738) — verification incomplete.** 13-config matched-`t` ledger complete on both arms, **no fate change on any config**, ΔR2_max 7.6–20.5%. **SDHS changed phase structure** (stock hands over at `t`=0.147/0.791; C3c stays energy-driven to 1.5) — a fate-only check does NOT catch this, and `compare_trajectories.py` cannot see it because it diffs the terminal fate, not the phase sequence. PRB's 5661% is a collapse-floor artifact (C3c *delays* collapse 56%). **Full `pytest` DISCHARGED 2026-08-16** (during Batch 8, which changed no `trinity/` source): **1085 passed, 3 failed**, 16 deselected, 605 s. The 3 red are exactly the ones predicted below and nothing else — `test_run_smoke.py`, `test_phase_boundary.py` (`cool_beta` measured **0.8783952818088819** vs golden 0.888197, matching the recorded 0.878395 to the digit) and `test_mu_audit_drift.py` (site count measured **5** vs 11, matching the recorded 11 → 5). Confirmed pre-existing by re-running the two fast ones **in isolation** from Batch 8's new test file (trinity leaks module-level global state in-process, so this was checked, not assumed). Still owed: D4 goldens with before/after table, fold-back notes; `test_run_smoke` is **not** on D4's list and needs its own sign-off. CHANGELOG landed in `3590c91d`. Both arms ran at a pre-`main` base; Batch 7 re-ran B3M on `main` and reproduced the row exactly, so main's other physics is neutral **for that config only**. ⚠️ **[B11 2026-08-18] The default suite now has FOUR reds on this branch, not three.** Full `pytest` at `59b51c43`: **1129 passed, 4 failed**, 16 deselected, 370 s. The new one is `test_gen_default_param.py::test_committed_file_is_byte_identical_to_render`, which is **not** P_HII-related and is **pre-existing** — verified by running it in a clean worktree at this branch's base `ef624195`, where it fails identically, i.e. it predates every B11 commit. B11 touched no `trinity/` source at all. Flagged, not fixed: it belongs to whoever owns `default.param` rendering | `data/b6_ledger.csv`, `data/b6c3c_walltimes.csv`, `data/b6stock_walltimes.csv` |
| 7 | 🟡 | 2026-08-16 | **G7.2 PASSES — the control fires, so the null is evidence.** `B3MW001` (`Lw × 0.01`, `Qi` untouched) breaks confinement in the **energy** phase: 78.4% HII-dominated, `ratio_max` **4.927** against a pre-registered `[1.5, 6.0]` and a point prediction of 3.01 from `Qi^0.5 Lw^−0.7 ρ^−0.3 t^−0.1`. **G7.1 holds on all 8 nominal-wind configs** — 100% confined in the energy phase across **five decades** of core density (1e2–1e6 cm⁻³), worst margin GMC `ratio_max` 0.173, i.e. 2.9× below the registered 0.5 bar. Energy phase *closed* on 6 of 9 (PL2/SDHS/BE still inside it — partial coverage, not a closed null); implicit/transition/momentum coverage still accumulating. Recomputation validated against the delivered branch on 231/231 B3M rows, `mismatch_rows`=0. Verdict: **`P_HII`≡0 in the energy phase is a property of the regime, not a theorem** — it survives ~1.5 decades of wind suppression and breaks at 2 | `data/b7_confinement_screen.csv`, `data/b7_regime_trajectory.csv`, `figures/b7_regime.png`, `figures/b7_feedback_compare.png` |
| 8 | ✅ | 2026-08-16 | **C3a IS the classical D-type pressure — the photo-only limit is exact.** No solver run; the shipped `get_phii_c3c` driven through the thin-shell momentum equation. Algebra gates exact to machine precision (Strömgren anchor 2.2e-16; the `mu_convert/mu_ion_shell` prefactor **is** the 2.2 particles per H nucleus, so `P_C3a(R_St) = n_tot k T = rho_0 c_i^2` — Spitzer's `2nkT` with He). Dynamics: index → **0.57124** vs 4/7 = 0.571429, and deviation from Hosokawa–Inutsuka **0.0000% over `R/R_St ∈ [2,50]`** on all 5 `(n_0, Qi)` combinations, while sitting **8.56%** above Spitzer against the analytic `(4/3)^{2/7}` = 8.55% — it lands on the momentum-equation closure, not the ram-balance one. ⚠️ **G8.4 FAILED as registered** (9.511% vs a 5% bar): I compared a from-rest integration against a closed form whose `t=0` state is `v = sqrt(4/3) c_i`, so the gate measured the startup transient (−9.51% at `R/R_St`=2 → −0.01% at 150, index converging on 4/7). Recorded as failed and **amended** (G8.4′), not reinterpreted; the amendment was checked not to weaken the gate — the mis-normalised control still misses by −20.14% vs analytic −20.17%, and the pinned tests fail on `P_C3a × 1.001`. ⚠️ **Not independent confirmation** — HI is *derived from* the same momentum equation, so once the algebra gates hold the ODE must return HI. The content is in G8.2 (the prefactor could have been 1, 2, or a `mu` confusion; it is the He-correct 2.2); the dynamics are a propagation + sensitivity check. **Consequence:** the shipped docstring's "NOT an O(1) normalisation error" is now externally anchored and **confirmed** — the universal HII-dominated momentum phase is not a prefactor bug, so re-deriving C3a's normalisation is a dead end; what stays open is the `R2^{−3/2}` vs `R2^{−2}` geometry, a model-structure question. Both halves of §3's limiting-case obligation on C3 are now discharged | `data/b8_spitzer_crosscheck.csv`, `harness/spitzer_crosscheck.py`, `test/test_phii_c3c_spitzer.py` |
| 9 | 🟡 | 2026-08-17 | **G9.2 FALSIFIED in momentum, G9.3 discharged — and my own scoping headline retracted.** One B3M run (`--arm b9`, code `2fa8cc9c`, clean tree) reproduced the known trajectory exactly (231 rows, 4 phases, `R2_end` 23.253 vs Batch 5's 23.25) and covered the momentum phase the scope could not. The geometry ratio `sqrt(R2/(3 dR))` crosses 1 at `dR = R2/3`, and the shell goes from thin (`dR/R2` ~1e-3, energy) to **thick** (0.670–1.308, momentum), crossing over *inside* transition. So `frac_ratio>1` = 1.0000 / 1.0000 / **0.3810** / **0.0000** across energy/implicit/transition/momentum. ⛔ **The scope's claim that the correction is one-signed and *raises* `P_HII` 1.75–100× is retracted** — in momentum it **lowers** it 0.51–0.71×, i.e. it moves the *helpful* direction, the opposite of what I reported before momentum was covered. ✅ **What survives:** it is not enough — `P_HII/P_ram` median 6.165 → layer-corrected **3.594**, **34/34 momentum rows still HII-dominated**, so a geometry fix cannot produce the wind-dominated branch and D5 (pressure coupling) stays live for a weaker reason than claimed. 🔍 Unregistered but notable: the layer-corrected ratio is nearly **time-independent** (3.584→3.614 over t 0.405→1.5) while the cavity form *climbs* 5.08→7.16 — the growing dominance is a geometry artifact. **G9.4 CLOSED same day: also FALSIFIED (3.171× vs a 2× bar)** — the analytic thin-layer scaling overestimates the real profile's recombination-equivalent density by up to 3.2×, and in the thin-layer phases the gap **is** exactly the dust sink (`sqrt(recomb/Qi_abs)` = 0.497/0.907 vs measured 0.496/0.906, three decimals). G9.2's momentum verdict survives the recheck with the true ionised thickness (0.505–0.712, 0/17 > 1; the shell is 99.54% ionised there, so the clamped `dR` was benign — now measured, not assumed). ⚠️ **But the profile form supersedes this row's 3.594**: the same rows give **median 1.545 (1.322–1.666), falling with time**, so the momentum excess over unity is ~50%, not ~260%. 🔍 Extrapolating stage 3's `Lw^−0.33` puts inversion at **`Lw ≈ 3.4`** rather than 260 — physical, and `B3MW3`/`B3MW10` already exist. Flagged as a lead needing its own measurement, not a result | `data/b9_geometry_scope.csv`, `data/b9_layer_density.csv`, `harness/geometry_screen.py`, `harness/layer_density_check.py` |
| 10 | ✅ | 2026-08-17 | **G10.2 FALSIFIED — the `Lw` ≈ 3.4 lead is dead, and D5 is back to being the route.** `B3MW3` + `B3MW10` ran clean to `stop_t` 1.5, both reached momentum, and their energy row counts (96/105) reproduce stage 3's exactly. Profile-form momentum medians **1.5451 → 1.3412 → 1.1902** for `Lw` = 1/3/10: G10.1 passes by 0.0088 (1.3412 vs a 1.35 ceiling, 25% above its 1.0752 point prediction), G10.3 (cavity form HII-dominated 100%, both rungs) and G10.4 (monotonic) pass, but **G10.2 fails — 1.1902 against a registered < 1.0**. The failure mechanism was registered in advance and is confirmed: stronger winds **thin the shell** (`dR_ion/R2` ∝ `Lw^−0.3375`), so the geometry correction `sqrt(R2/3dR)` *rises* as `Lw^+0.169` and cancels ~43% of the cavity form's `Lw^−0.3959` decline. Net profile response is only **`Lw^−0.1133`**, not the −0.33 my Batch 9 extrapolation assumed. Revised inversion: cavity ≈ `Lw` 99, profile ≈ `Lw` **46.5** — a ~2× improvement, not the ~26× projected, and **still unphysical**. 🔍 Genuine refinement: `frac(ratio>1)` in momentum is 0.0000/0.0000/**0.1667** because `B3MW10` dips to `dR_ion/R2` = 0.3197, below the `R2/3` break-even — so **the geometry correction's sign is wind-dependent**, which reconciles the Batch 9 scope ("raises") with the Batch 9 verdict ("lowers"): both were partial views of a thickness-dependent sign. ⚠️ G10.5 binds — the rungs sit at 0.98/0.74/0.45 thickness, so the trend is confounded with geometry and −0.1133 is not a clean wind response | `data/b10_wind_profile.csv`, `data/b10_walltimes.csv` |
| 11 | 🟡 | 2026-08-18 | **B11.0 DONE — the audit survives an adversarial pass: 3 CONFIRMED, 1 REVISED, 0 REFUTED. A–D not started.** One fresh B3M run at `ef624195` (495.9 s, 231 rows, all 4 phases, `R2_end` 23.253) reproduces `data/b9_layer_density.csv` to **≤3.3e-06 rel on 116 rows × 15 numeric columns**, so the Batch 9/10 baselines are sound and the seams are not a run artefact. **A CONFIRMED:** every `Qi` consumer in `trinity/` enumerated — no cavity-absorption factor at any of them, and the only attenuation (`get_shellODE.py:120`) starts at `r = R2` with `phi0 = 1`; `f_abs` = 1.0000 on 16/16 transition + 13/17 momentum driving rows, so the claimed budget is ≈`2·Qi`. Sharper than §6b put it: `f_abs` is *by construction* the shell's absorbed fraction and `get_bubbleParams.py:358` credits the **cavity** with that identical sub-budget. **B REVISED:** mismatch confirmed and sized (`P_HII/Pb` median 6.1646 momentum, 4.6218 transition; call-time ordering checked in all four runners) but the **direction is wrong** — feedback into `P_C3a` is **exactly zero on 29/33 driving rows** (`f_abs` already saturated) and **upward** on the other 4, so B is not an upper-bound mechanism, and §6b's "every seam pushes the same way" is corrected in place. **C CONFIRMED to 4 s.f. and understated:** `units-reviewer` cleared the derivation (`shell_structure.py:125-126` *is* the shipped inverse); route-P (invert `P_HII`) / route-Q (replay the forward map) agree to **1.000000000000** on all 33 driving rows; the committed `n_cavity` column gives 57,396.6 Msun by a third route. Re-measured **0.0952 (t=0.4074) → 0.5638 (t=1.5), 57,397 vs 101,805 Msun** vs the audit's 0.095 → 0.564, 57,400 vs 101,800. **New:** `shell_mass/M_avail` = 0.999997–1.000000 on every driving row — the shell already holds 100% of the gas that exists — and winds inject only **54.8 Msun** (0→1.5 Myr, from the run's own feedback columns; `bubble_mass` is *frozen* at 99.643 through momentum and unusable). So `(M_cav + M_shell)/M_avail` = **1.5638**: the model asserts 56% more gas than the run has, and over-subscription starts at the **first driving row** (t = 0.3037, transition), not in momentum. **D CONFIRMED:** `dR_full/R2` 0.6723–1.3078, `dR_ion/R2` 0.6579–1.3076, `dR_ion/dR_full` median 0.9954. **Side-findings S1–S4** (outside the seams, they change committed numbers): S1 `layer_density_check.py:140`'s thin-shell `V_lay` overstates `n_layer_analytic`/`ratio_analytic`/`pdrive_analytic` by 1.342–1.696× in momentum (`V_exact/V_thin` 1.802–2.878) — G9.2/G9.4/G10.1–4 verdicts all survive, and so do Batch 10's `Lw^−0.1133` fit and `Lw ≈ 46.5` inversion (both fitted to `pdrive_profile`, which never uses `V_lay` — re-derived from the published medians to check), but Batch 10's "flat to 1–3% within each rung" is withdrawn and the affected columns are re-fit in **B11.F**; S2 `layer_density_check.py:154`'s falsy-zero filter drops `P_HII == 0.0` rows (0/17 in momentum, so no published number affected); S3 `run_momentum_phase.py:888-896` never recomputes `P_HII` (no effect here — proved by the route agreement); S4 `b9_walltimes.csv` gains a real B3M timing, 495.9 s vs the mtime-derived ~590 s. **External:** all four seams are absent by construction from Geen et al.'s two-equation closure — registered as **B11.G**. — **B11.A–D DONE the same day, and two pre-registered gates came back against §6b.** **G11.A1:** the photon-conserving fixed point `x = f_abs(Qi(1−x))` has the unique root **`x = 1` on 33/33 driving rows, no interior root** — the cavity takes every photon and the shell is left neutral, so C3a cannot be closed photon-conservingly without a second equation. **G11.A2 REFUTES §6b seam A's consequence clause:** `P_C3a_fixedpoint/P_C3a_shipped` = **1.0000–1.1778, 0/33 below 1**, where "⇒ `P_C3a` overstated" predicted < 1 — conserving photons *raises* it by up to 17.8%. **G11.B1/B2:** both falsifiers of B11.0's seam-B revision failed to fire (0 non-zero Δ`f_abs` on 29 saturated rows; Δ`P_C3a` = +0.0000…+0.1778, 0 negative), and the inconsistency is large — `shell_n0` ×4.70/×6.17, ionised layer thins 79–83%, dust fraction 0.620→0.455 and 0.607→0.395, so seam B and G9.4's dust are **not independent**. **G11.C1:** photoevaporative supply 5.10e4–8.68e4 vs required 2.40e4–6.31e4 Msun/Myr ⇒ ratio **1.32–2.13 on 100% of rows**, so the cavity is **not rate-limited** — which *closes* §6b's "supply-limited" limb rather than supporting it, while B11.0's "no reservoir" result stands. **G11.C2a control PASSES at 0.871%** (2% blocking bar), so **G11.C2b** is a measurement: debiting the shell by `M_cav(t)` gives `R2(t=1.5)` **+8.55%** (inertia) / **+9.22%** (inertia+gravity). ⚠️ My own pre-registered "order 10–30%" **missed** — sign right, magnitude ~9% — recorded as a miss because `M_cav/M_shell` is only 0.10 for most of the momentum phase. **Net: §6b's "upper bound" list loses both A and B and is down to seam C + G9.4's dust** | `data/b11_mass_ledger.csv`, `data/b11_photon_ledger.csv`, `data/b11_mass_dynamics.csv`, `harness/mass_ledger_check.py`, `harness/photon_ledger.py`, `harness/mass_ledger_dynamics.py`, `data/b9_walltimes.csv` |
| 12 | ✅ | 2026-08-18 | **The low-wind sanity check the workstream was missing — and it fired one of my own falsifiers.** Both arms of `B3MW01` (`Lw × 0.1`) ran to `t` = 1.5 in **momentum** (c3c 702 s/205 snaps, stock 666 s/222 snaps, fate unchanged), so nothing is VOID. **G12.1 PASS both halves:** the OLD identity reproduces out-of-sample at a wind strength Batch 0 never ran — `\|P_HII/Pb − 1\|` ≤ **4.44e-16**, frac within 1e-12 = **1.0000** in implicit/transition/momentum (energy 0.9855, the documented stale-`Pb` handoff rows) — while the NEW code gives `P_HII` = 0.0 on 69/69 energy + 79/79 implicit and `P_HII/Pb` = 1.240–14.369 with **0 rows ≤ 1** on the driving branch. **This is the cleanest old-vs-new demonstration in the workstream:** cut the wind 10× and the old code still returns the confining pressure to 2.2e-16 (it carries no photoionisation information at *any* wind strength) while the new one runs 100% HII-dominated at `P_HII/Pb` ≈ 14 and reaches `R2` = 7.733 pc vs 5.722 pc. **G12.2 PASS + validates the offline screen a decade below where it was tested:** measured `frac_HII_dom` transition **0.9062** / momentum **1.0000** against the screen's 0.9118/1.0000, despite the arm's `R2` being 35% larger; momentum `P_HII/Pb` is **2.2×** the nominal rung, consistent with stage 3's `Lw^−0.33`. **G12.4 prediction held:** ΔR2 = **+35.138%** at matched `t`, against B3M's own 20.517% and the whole 13-config b6 range of 7.6–20.5%. ⛔ **G12.3 fired my registered seam-C falsifier:** `M_cav/M_shell` = **0.1296** < the 0.2 bar, so **§6b's 0.564 is a B3M number, not universal** — `M_cav ∝ R2^{3/2}` makes the seam track **bubble size, not HII dominance**, and the low-wind run only reaches 7.733 pc. My stated reasoning was wrong; the bar stands as written. **The qualitative core survives:** still over-subscribed at `(M_cav+M_shell)/M_avail` = **1.1296**, wind mass 31.6 Msun = 0.24% of `M_cav`, dynamical cost +0.446%/+0.965% (control 0.213%). **Seams A/B/D do NOT follow C:** A generalises exactly (27/27 endpoint `x*`=1, ratio 1.0000–1.0000), B holds *and doubles* (`shell_n0` ratio 11.84/13.77 vs 4.70/6.17; layer thins 88–90%; the lone negative Δ`P_C3a` is **−1.46e-16 = 0.66 ULP** with Δ`f_abs` identically 0 — roundoff, and the harness now says so instead of printing a trip), and D is **worse** (`dR_ion/R2` 1.171–1.438 vs 0.658–1.308) | `data/b12_lowwind_trajectory.csv`, `data/b12_lowwind_mass_ledger.csv`, `data/b12_lowwind_photon_ledger.csv`, `data/b12_lowwind_mass_dynamics.csv` |
| 14 | 🟡 | 2026-08-28 | **Offline screen DONE; no arm (SHIP-HOLD).** G14.0 passes by the letter (no linear slope in [0.95,1.05] with r²>0.99) but the disclosed log-log diagnostic shows both variants ∝`Pb^1.0` on driving rows (K5a r² 0.988 gain ≈2.4, K5b r² 0.993 gain ≈1.5) — the identity with an O(1) gain. K5a identified as the code's own uncapped `n_IF_Str_raw`; Batch 3b had already measured that coupling at r ≥ 0.997 on 788 pre-C3c rows, and Batch 4a its full-run cost (ΔR2 15.3–28.4%, ramp-window concentrated). Branch census: 83/83 confined B3M rows flip driving under both variants (K5b = 1.073×`Pb` in energy — the old identity to 7%), first flip inside `dt_switchon`. G14.3: momentum `P_HII/P_ram` 6.165 → 2.381 (K5a) → 1.545 (K5b), B3MW01 13.766 → 4.388; both gate-text priors hit. Verdict: the cavity volume IS the decoupling; the layer volume belongs in the coupled closure (K10/K6). Arm scope (bare swap / driving-only / defer to K10) is the maintainer's call | `data/b14_k5_screen.csv`, `harness/k5_offline_screen.py` |
| 16 | ✅ | 2026-08-28 | **K10's composition mapping SOLVED and gated through the real `P_drive` expressions.** G16.0 PASS (worst 2.22e-16 vs 1e-12, all four phases, both `Q_eff` variants): `return = P_conf·ρ − (P_ram if the composition adds it)` yields the CEM drive exactly. G16.1 PASS (853/853 non-negative). G16.2 PASS — the confined-limit term is delivered at **+0.96%** median over `P_conf` instead of being swallowed by the `max`, which was Batch 14's specific finding. G16.3 established the predicted signature change: inside `dt_switchon` the ramped/un-ramped `P_conf` ratio is 0.33–0.995 (median 0.711), so K10 must receive the **ramped** `press_bubble`, which `params` does not carry. G16.4 reproduces Batch 13's magnitudes exactly by a different code path (momentum 1.605/0.851 B3M, 2.242/1.096 B3MW01) — an independent cross-check. Remaining K10 blockers: dust (Batch 17) and a full-run arm | `data/b16_composition.csv`, `harness/k10_composition_check.py` |
| 17 | ✅ | 2026-08-28 | **Dust is now INSIDE the closure and validated against trinity's own shell solve — G13.4's blocker discharged.** The closure is `get_shellODE.py:120`'s own photon equation at uniform `n₀`, so it is a reduction of the code's dust treatment, not a new model. **G17.0 PASS**: predicted/measured `f_dust` median **1.056**, 97.3% of rows within 25%, both configs agreeing (1.064 / 1.052) — the G9.4 risk did not materialise because dust is linear in `n` where recombination is quadratic. G17.1 PASS (436/436 fronts). G17.2 PASS (σ_d→0 recovers the closed form to 9.3e-13). ⛔ **G17.3's pre-registered expectation MISSED**: the self-consistent drive is not between the no-dust and post-hoc values but just below both (`c/b` 0.92–0.95) — the closure debits photons continuously rather than once. Re-run sensitivity is still ≈2× (1.886–2.214), which is dust genuinely mattering, not a revived blocker. G17.4 end-to-end: composed/shipped 1.006/1.005/0.746/0.884 (B3M) and 1.059/1.099/0.982/1.034 (B3MW01). ⚠️ ED-phase dust validated on **1 row** — the ledgers only replayed driving rows | `data/b17_dust_closure.csv`, `harness/k10_dust_closure.py` |
| 18 | ⛔ | 2026-08-28 | **HELD by Batch 20 slice 2, then SUPERSEDED by Batch 21's O1 form.** K10 implemented as an arm and per-call gated; the ladder is owed **against `k10_o1_arm.patch`, not this one**. Patch adds `get_phii_k10` + `_k10_front_radius`, aliases `get_phii_c3c` to it, **zero call-site edits**, no new dependency. ⛔ **G18.0 FAILED as written (6.761e-02 vs 1e-10)** — recorded, not re-barred; diagnosed as *entirely* the `P_conf` source (`P_conf` rel err **0.0 in implicit and momentum**, 6.8e-2 energy / 5.9e-3 transition): the screens recovered it from stored columns, production recomputes it from `Eb` and a solved `R1`, and **production is correct**. ✅ **G18.0′ amendment PASSES at 1.005e-12** with `P_conf` held fixed, so the production closure IS what Batches 16/17 validated. ✅ Arm runs clean (SC, 114 snapshots, 338 s, zero distress). ✅ G18.1's contract change live: `P_HII > 0` on 97/97 energy and 17/17 implicit rows where C3c gives exactly 0.0. ⚠️ Bounds Batch 17's offline error: momentum/implicit exact, transition ≤0.59%, energy ≤6.8% **on 2 of 156 rows only** (narrowed 2026-08-29) | `data/b18_percall.csv`, `harness/k10_percall_equivalence.py`, `hpc/b14/k10_arm.patch` |
| 21 | ✅ | 2026-08-29 | **K10-O1: read the shell solve's own front instead of solving one — three of Batch 20's five blockers closed.** G21.0 PASS (`R_IF` read verbatim; `_k10_front_radius` **deleted**, so slice 1's F1/F2/F4 close by deletion). G21.1 PASS — front inside the shell on **140/140** rows, against 18/18 momentum + 43/44 energy violating before. G21.2 PASS — implied layer mass ≤ shell mass on every row (max 0.94 vs 2.49 before), so **seam C is closed**. ⚠️ G21.3's disclosed prior missed by 16% (3.274 measured vs 3.901 registered; row-set difference, recorded). Confined branch left alone (O1/shipped 1.001/1.005); driving branch **halved** vs shipped (×0.494 B3M momentum, ×0.325 B3MW01). G21.4 — the confined excess survives at **+0.475%**, about half the Batch 18 form, so Batch 16's first-order-term argument stands reduced. G21.5 verified under the patch: **still no photo-only limit** (spitzer 5 failed), freeze ratchet untouched, coverage unchanged | `data/b21_o1_screen.csv`, `harness/k10_o1_screen.py`, `hpc/b14/k10_o1_arm.patch` |
| 13 | ✅ | 2026-08-27 | **K10 screened offline; two of my own gates failed by design error and the dust rule fired.** State-jump exactly 0 where the shipped rule jumps +34% at the first post-crossing snapshot (the scheme's intrinsic discontinuity is `P_ram/Pb` = 23.4% — corrected 2026-08-29); healthy branch +0.68% on B3M; **G13.4 dust rule fired at 2.05× ⇒ K10 cannot ship without a dust model** (discharged by Batch 17, then made moot by Batch 21's O1). G13.1/G13.3 FAILED and are recorded with diagnoses; G13.3 yielded `R_ch`(trinity) = `chi_e`·`R_ch`(Lancaster) | `data/b13_k10_screen.csv`, `harness/k10_cem_drive_screen.py` |
| 20 | ⛔ | 2026-08-29 | **K10 safety audit — UNSAFE as implemented; the arm was HELD on this.** Four adversarial slices, registered before any reported. **4 MAJOR + 1 MODERATE**: no photo-only limit (`test_phii_c3c_spitzer.py` 6 passed → 5 failed; re-labelled from CRITICAL-runtime to MAJOR-domain on the maintainer's challenge, then measured stronger — front outside the cloud on **100% of driving rows**, max 72.7 pc in a 5.0 pc cloud); **seam C present and worse than shipped** (2.4892× the shell); the **per-segment freeze ratchet** re-armed (~8% median, 17% max staircase vs the 0.55% term it delivers); coverage 2 configs of 13; and `test_mu_audit_drift.py` passing **vacuously**. ✅ Survived: seam A absent, composition mapping (2.22e-16) with non-negativity **proven**, dust closure validated, G13.3's `χ_e` diagnosis confirmed and strengthened, no jump at the transition→momentum handover. ✏️ Two of the auditor's own claims refuted (the `n_IF_Str` gate: 0/3490 rows; the thin-dust guard: tests the wrong quantity, never fires) | `data/b20_domain.csv`, `harness/k10_domain_check.py`, `data/b22_bubble_density.csv`, `harness/bubble_density_probe.py` |
| GB | ✅ | 2026-08-29 | **`bubble_mass` freeze FIXED in production (momentum only) — the session's only `trinity/` behaviour change.** The bubble solve runs only in energy/implicit, so `bubble_mass` was a stale carry-over: 99.6429 Msun against a true enclosed mass of 0.116 (**860× too large**). Fixed with `mass_freeWind` = `2·L_mech·R2/v³`, reusing `pRam`'s convention. **All five gates pass**; GB.3 is **bit-identical** (`dR2_max` 0.000%, **0 of 220 rows** differ, `R2_end` 14.0584340349 both arms), because `bubble_mass` reaches only `shell_grav_*`, which is diagnostic-only and never enters the EOM. Transition deliberately left stale and marked in-source | `data/gb3_bubblemass_ledger.csv` |
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
| `data/b11_photon_ledger.csv` | `harness/photon_ledger.py` | 11 (B11.A/B) | same run; ~9 shell solves per row, ~2 min |
| `data/b11_mass_dynamics.csv` | `harness/mass_ledger_dynamics.py` | 11 (B11.C) | same run; offline LSODA re-integration of the momentum EOM |
| `data/b12_lowwind_trajectory.csv` | `harness/reduce_phii_regime.py` | 12 | **both arms**: stock = worktree @ `fca7d88e` (pre-C3c), c3c = `bac9547e`; `B3MW01`, `stop_t` 1.5 |
| `data/b12_lowwind_mass_ledger.csv` | `harness/mass_ledger_check.py` | 12 | c3c arm only |
| `data/b12_lowwind_photon_ledger.csv` | `harness/photon_ledger.py` | 12 | c3c arm only |
| `data/b12_lowwind_mass_dynamics.csv` | `harness/mass_ledger_dynamics.py` | 12 | c3c arm only |
| `data/b11lowwind_walltimes.csv` | `harness/run_batch.py` | 12 | c3c arm timing (702 s / 205 snapshots) |
| `data/b11g_cem_closure_check.csv` | `harness/cem_closure_check.py` | 11 (B11.G rung 0) | no run — scale-free numeric verification of the two CEM identities K6 leans on |
| `data/b15_ionised_mass_fraction.csv` | `harness/ionised_mass_fraction.py` | 15 (G15.0) | replays `shell_structure_pure` and integrates the profile; B3M + B3MW01 |
| `data/b13_k10_screen.csv` | `harness/k10_cem_drive_screen.py` | 13 | no run — K10 vs shipped drive on committed B3M + B3MW01 trajectories, dust variant joined from the photon ledgers |
| `data/b14_k5_screen.csv` | `harness/k5_offline_screen.py` | 14 (offline) | no run — K5a/K5b vs shipped on committed CSVs; B3M = b9_layer × b11_ledger row_idx join, B3MW01 = b12 photon-ledger driving rows (K5a only) |
| `data/b14_identity_census.csv` | `harness/identity_census.py` | 14 (maintainer Q) | no run — per-phase identity + drive-composition census, stock vs c3c, from the committed both-arm trajectories (b7 B3M, b12 B3MW01) |
| `data/b14_cavity_gas.csv` | `harness/cavity_gas_check.py` | 14 (maintainer Q) | no run — cavity gas content vs C3a's asserted photon sink, from the B3M/B3MW01 mass ledgers |
| `data/b16_composition.csv` | `harness/k10_composition_check.py` | 16 | no run — K10 mapping routed through the real `P_drive` expressions on committed c3c trajectories, both `Q_eff` variants |
| `data/b17_dust_closure.csv` | `harness/k10_dust_closure.py` | 17 | no run — dust inside the K10 closure (uniform-`n₀` reduction of `get_shellODE.py:120`), validated against the ledgers' measured `dust_Pb` |
| `data/b18_percall.csv` | `harness/k10_percall_equivalence.py` | 18 | no run — implemented `get_phii_k10` vs the Batch 17 screened closure; carries the `P_conf` diagnostic that explains G18.0's failure |
| `hpc/b14/k10_arm.patch` | the exact Batch 18 arm diff (apply in a detached worktree) | 18 | arm code; never merged — D5 open |
| `data/b21_o1_screen.csv` | `harness/k10_o1_screen.py` | 21 | no run — O1 (`ρ` from the shell solve's `R_IF`) vs the Batch 18 form on the same committed rows |
| `hpc/b14/k10_o1_arm.patch` | the Batch 21 O1 arm diff; **supersedes `k10_arm.patch`** | 21 | arm code; never merged — D5 open |
| `data/b20_domain.csv` | `harness/k10_domain_check.py` | 20 | no run — is K10 outside its own domain inside trinity's regime? |
| `data/b22_bubble_density.csv` | `harness/bubble_density_probe.py` | 20 (exploratory) | no run — bubble real density from mass/volume; closed as unusable for `P_HII` |
| `data/gb3_bubblemass_ledger.csv` | `harness/compare_trajectories.py` | GB | matched-`t` ledger for the `bubble_mass` fix; bit-identical |
| `LITERATURE_ASSESSMENT.md` | external input (not authored here) | — | **C-0 carve-out**, rev2 2026-08-18. Never load-bearing: cite for attribution only. *Indexed here 2026-08-29 — C-0 condition 4 had been unmet since the carve-out was written* |
| `data/b19_cancellation.csv` | `harness/cancellation_check.py` | 13 (correction; the `b19_` prefix is a naming wart — it belongs to the Batch 13 retraction, not to a Batch 19) | no run — factorial test of the volume/dust corrections to C3a on committed B3M rows; retracts the cancellation claim |

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

- **2026-08-18 (Batch 11, B11.A–D — the seams are quantified, and two of my own registered
  predictions lost)** — Ran the three measurement batches against gates written and committed
  *before* any of them executed. The gates did their job: three of them came back against what was
  written, and all three are recorded as misses rather than reinterpreted.
  **B11.A. The fixed point is degenerate, and repairing the double-spend RAISES `P_C3a`.** A cavity
  Strömgren-filled at `n(x) = sqrt(3 x Qi/(4πχαR2³))` consumes exactly `x·Qi` for *any* `x`, so the
  cavity balance is one equation in two unknowns; the shipped code closes it by fiat with
  `x = f_abs(Qi)`. The photon-conserving closure `x = f_abs(Qi(1−x))` has the unique root **`x = 1`
  on all 33 driving rows, with no interior root anywhere** — the cavity absorbs every photon and the
  shell is left **neutral**, contradicting trinity's own 99.5%-ionised momentum shell and the
  boundary condition that sets `nShell0`. ⛔ And `P_C3a_fixedpoint/P_C3a_shipped` = **1.0000–1.1778,
  0 of 33 rows below 1**, where §6b's seam-A clause ("less than `Qi` available ⇒ `P_C3a` overstated")
  predicted below 1. **That clause is struck.** The double-spend is still real — B11.0 confirmed it —
  but it is not a reason the pressure is too high, and seam A leaves the "upper bound" list.
  **B11.B. B11.0's revision survives its own falsifiers.** Neither registered falsifier fired:
  `Δf_abs` = 0 on all 29 saturated rows, and `ΔP_C3a/P_C3a` = +0.0000…+0.1778 with zero negative
  rows. Descriptively the inconsistency is large — `shell_n0` rises 4.70× (transition) / 6.17×
  (momentum), the ionised layer thins by 79–83%, and the dust-absorbed fraction of ionising photons
  moves 0.620 → 0.455 and 0.607 → 0.395. That last number matters beyond seam B: G9.4's dust sink is
  itself pressure-dependent, so **seam B and the dust finding are not independent** and should not be
  added as if they were.
  **B11.C. The rate is fine, the reservoir is not, and it is worth ~9%.** Photoevaporation off the
  shell's ionised face supplies 5.10e4–8.68e4 Msun/Myr against a required `dM_cav/dt` of
  2.40e4–6.31e4 — ratio **1.32–2.13 on 100% of rows**, so the cavity is **not** rate-limited. That
  *closes* §6b's "supply-limited" limb rather than supporting it; what remains is B11.0's harder
  result that there is no reservoir at all, and a real flow at this rate is simply the double-book
  restated as a flux. The blocking control gate passed at **0.871%** against a 2% bar (offline
  23.0503 pc vs the run's 23.2527 pc at t=1.5), so the debited integration is a measurement and not
  VOID: debiting the shell by `M_cav(t)` gives `R2(t=1.5)` **+8.55%** (inertia only) or **+9.22%**
  (inertia and gravity). ⚠️ **My pre-registered "order 10–30%" missed.** The sign held, the magnitude
  did not, and it is recorded as a miss. The reason is legible in the data: inertia enters as `1/m`
  and `M_cav/M_shell` is only 0.10 for most of the momentum phase, reaching 0.56 only at the end.
  So "does 56% mass matter?" has a number — **~9% in `R2`**, comparable to C1's 4.0% and well below
  the 12.8–20.5% C3c itself moved.
  **B11.D.** Stated as a validity limit rather than a defect: the momentum-phase ODE assumes a thin
  shell and C3a assumes a sharp cavity/shell split, and on B3M's momentum rows `dR/R2` = 0.66–1.31
  with the shell 99.54% ionised, so neither premise holds and "cavity" and "shell" are not
  distinguishable by ionisation state there.
  **Where this leaves §6b.** All four seams exist, but the summary claim that they all push the same
  way has now lost **two of its three members** — A's repair raises `P_C3a` by up to 17.8%, and B's
  is zero-or-up. Only **seam C** (the filled-cavity limb needs 56% more gas than exists) and
  **G9.4's dust** bound the driving-branch pressure from above. Both direction claims were struck by
  measurement, in the same way seam B's was during B11.0 — which is the argument for having
  pre-registered them.
  **Where this leaves D5.** The load-bearing result is G11.A1, not any of the magnitudes: C3a's
  photon-conserving fixed point is degenerate, so seams A and B **cannot be repaired inside C3a's
  structure**. D5's question therefore moves from "C3c-switch vs C3a-raw" to "C3a at all", and
  **B11.G** — scoring the shipped closure against Geen et al.'s two-equation treatment, which has all
  four seams absent by construction — becomes the cheap next step rather than an optional extra.
  Artifacts: `data/b11_photon_ledger.csv` + `harness/photon_ledger.py` (B11.A/B),
  `data/b11_mass_dynamics.csv` + `harness/mass_ledger_dynamics.py` (B11.C). No `trinity/` source
  touched.

- **2026-08-18 (Batch 12 — the low-wind rung; the sanity check lands, and my own falsifier fires)** —
  Maintainer asked whether the committed data covered a low-wind regime where `P_HII` must dominate,
  as an old-vs-new sanity check. It did not: the only paired stock-vs-C3c trajectory was `B3M`
  (`Lw` = 1), the replay family was `Lw` ∈ {1, 3, 10}, and every Batch 11 diagnostic was `B3M` alone.
  The low-wind material that existed was either an offline screen on *stock* trajectories (`B3MW01`
  in `data/b5s3_*`) or a run that never reached the driving phases (`B3MW001`, `run_complete=False`).
  Ran both arms of `B3MW01` (`Lw × 0.1`), gates registered and pushed before either finished.
  **The check itself is the cleanest old-vs-new evidence this workstream has.** Cut the wind by a
  decade and the old code *still* returns the confining pressure to **2.2e-16** — it carries no
  photoionisation information at any wind strength, which is the whole defect — while the new code
  runs **100% HII-dominated** in momentum at `P_HII/Pb` = 13.667–14.369 and reaches `R2` = 7.733 pc
  against the old 5.722 pc, i.e. **+35.138%** at matched `t`, above B3M's own 20.5% and above the
  entire 13-config b6 range. G12.1 also gave the identity its **first out-of-sample confirmation** —
  Batch 0 measured it on six configs but never at this wind strength — reproducing to the same 2e-16,
  with the one exclusion being the already-documented stale-`Pb` rows at the 1a→1b handoff.
  **The offline screen survived a real test.** `c3_offline_screen.py` predicted `frac_HII_dom`
  transition 0.9118 / momentum 1.0000 from the *stock* trajectory; the real C3c arm measured
  **0.9062 / 1.0000** even though its `R2` ends 35% larger. That screen had only ever been validated
  at nominal wind (Batch 5 stage 2); it now holds a decade below, which materially raises how much
  weight offline screens can carry in this workstream.
  ⛔ **And it fired the falsifier I registered against seam C.** I wrote that
  `M_cav/M_shell < 0.2` at t=1.5 "would make the mass double-book regime-specific and materially
  weaken §6b seam C". Measured **0.1296**. Recorded as fired, bar unchanged. **My reasoning was the
  wrong way round**: I argued a weaker wind hands more of the drive to the HII term so the seam must
  worsen, but `M_cav ∝ R2^{3/2}·sqrt(Qi f_abs)` makes the controlling variable **bubble size**, and
  the low-wind run only reaches 7.733 pc against 23.253 pc — the `R2^{3/2}` ratio is 0.192 against a
  measured mass ratio of 0.226. So the mass double-book is worst in the configs that **expand
  furthest**, not the ones where photoionisation dominates most, and §6b's 0.564 is a **B3M number**
  that must always be quoted with its config. What survives untouched is the load-bearing half:
  still over-subscribed at `(M_cav+M_shell)/M_avail` = 1.1296, wind mass 31.6 Msun = 0.24% of
  `M_cav`, so there is still no supply — just less of it to not-supply. The dynamical cost falls with
  the mass, to +0.446%/+0.965% against a control passing at 0.213%.
  **The other three seams do not follow C, which is the structural result.** A generalises exactly
  (27/27 driving rows give the endpoint root `x*` = 1, and the fixed-point ratio is 1.0000–1.0000
  because `f_abs` is saturated on every driving row); B holds *and roughly doubles* (`shell_n0` ratio
  11.84/13.77 against 4.70/6.17, layer thinning 88–90% against 79–83%); and D is **worse**
  (`dR_ion/R2` = 1.171–1.438 against 0.658–1.308). So only seam C is regime-scoped.
  ⚠️ **One harness defect found and fixed while grading G11.B2.** The gate printed
  `rows < 0: 1/27`, which reads as a falsifier trip. The row's Δ`P_C3a` is **−1.459e-16 = 0.66 ULP**
  on a row whose Δ`f_abs` is identically 0 — roundoff in recomputing `P_C3a` through a `sqrt`, not a
  sign reversal. `photon_ledger.py` now counts real negatives and roundoff separately against a
  1e-14 floor and prints both, so a future reader is not misled by its own gate. The CSVs keep the
  raw signed values either way.
  Artifacts: `data/b12_lowwind_trajectory.csv` (both arms, per-row), plus the three B11 harnesses
  re-run on the c3c arm. No `trinity/` source touched.

- **2026-08-18 (reconcile pass — §0 amended, D5 register built, and §9's Batch 7–10 gap backfilled)** —
  Maintainer asked which document is the source of truth and whether the influx of paper `.tex`
  files and an external assessment warranted a reconvene. Both concerns were correct.
  **(1) §0 amended.** The source of truth is this file; `README.md` is the pre-C3c evidence record;
  `DOC_STATUS.md` tracks workstreams. Adding `LITERATURE_ASSESSMENT.md` on 2026-08-18 **violated
  §0's "one doc" rule** and I did not flag the conflict at the time. Maintainer ruled it stays, so
  §0 now carries an explicit **C-0 external-document carve-out** with five conditions — not
  authored here and kept verbatim, a provenance block, a cross-check section written by us,
  indexed in §8.3, and **never load-bearing**. The last condition is the important one: no ledger
  verdict and no gate may cite an external document as evidence. Exercised by exactly one file.
  **(2) The assessment is marked pending correction.** The maintainer flagged it "might contain
  wrong information" and is preparing an update, so its §4 is now labelled unverified motivation.
  Blast radius checked rather than assumed: **no measurement depends on it.** B11.0, B11.A–D and
  B12 all derive from source read here and runs done here; the four-source cavity-volume result was
  verified by grepping the `.tex` files directly. Exactly four PLAN.md lines cite it (1717 and Batch
  12's rationale), all attribution or motivation, so nothing measured moves if §4 is wrong.
  **(3) §7.1 D5 candidate register added.** D5's cell had grown by accretion over ~six sessions and
  Batches 9–12 plus the assessment added four more candidates in scattered prose. There is now one
  table, K1–K9, with evidence tiers (measured / source-supported / from the pending assessment /
  unmeasured) and an explicit note on how the rows relate — K1/K2 are branch logic, K4/K5 change the
  value, K6 subsumes K5, K7–K9 are orthogonal. Any new candidate gets a row or it does not exist.
  Recorded there and repeated here because it is the re-ranking §0 step 4 requires: **B11.A's
  degeneracy result argues against the K1/K2 family as a whole**, not just against one branch rule,
  and points at K6 — but K6 is the biggest change on the table and B11.G is the cheap way to test it
  first. No decision taken; D5 remains the maintainer's.
  **(4) Batches 7–10 were never logged here.** §9 jumps from 2026-08-14 to 2026-08-18 while four
  batches ran, so §0's step-3 rule ("no entry, no edit") was not honoured at the time. Rather than
  insert back-dated entries — which would misrepresent the record — the gap is closed with the
  reconstruction below, **derived from the §8.1 ledger and the batch sections, not from
  contemporaneous notes**, and marked as such. Treat it as an index into those rows, not as new
  evidence.
  > **Backfill (reconstructed 2026-08-18 from §8.1; not contemporaneous).**
  > **Batch 7 (2026-08-16) — the confinement null is a regime property, not a theorem.** G7.1 held
  > on all 8 nominal-wind configs: 100% confined in the energy phase across five decades of `nCore`,
  > worst margin GMC `ratio_max` 0.173 against a 0.5 bar. G7.2, the control, **fired as designed** —
  > `B3MW001` (`Lw × 0.01`, `Qi` untouched) broke confinement in the energy phase at 78.4%
  > HII-dominated, `ratio_max` 4.927 against a pre-registered `[1.5, 6.0]` and a point prediction of
  > 3.01. Because the control fires, the null is evidence rather than an artefact. Recomputation
  > validated against the delivered branch on 231/231 B3M rows, `mismatch_rows` = 0.
  > **Batch 8 (2026-08-16) — the photo-only limit is exact, and it kills the calibration
  > explanation.** No solver run. C3a driven through the thin-shell momentum equation reproduces
  > Hosokawa–Inutsuka to **0.0000%** over `R/R_St ∈ [2,50]` on all five `(n_0, Qi)` combinations,
  > with index 0.57124 against 4/7, and sits 8.56% above Spitzer against an analytic 8.55%. The
  > `mu_convert/mu_ion_shell` prefactor **is** the He-correct 2.2 particles per H nucleus.
  > ⚠️ **G8.4 failed as registered** (9.511% against a 5% bar) because it compared a from-rest
  > integration against a closed form whose `t=0` state is already moving; recorded as failed and
  > amended to G8.4′ rather than reinterpreted. ⚠️ Not independent confirmation — HI is derived from
  > the same momentum equation. Consequence: re-deriving C3a's normalisation is a dead end.
  > **Batch 9 (2026-08-17) — geometry is not the escape hatch either, and the scope's headline was
  > retracted.** The scope claimed the geometry correction was one-signed and *raised* `P_HII`
  > 1.75–100×; the B3M momentum run **falsified that in the momentum phase**, where the shell is
  > thick and the correction *lowers* `P_HII` 0.51–0.71×. `frac_ratio>1` = 1.0000 / 1.0000 / 0.3810 /
  > 0.0000 across the four phases. G9.4 closed the same day and was **also falsified** (3.171 against
  > a 2× bar): the analytic thin-layer scaling overestimates the real profile's
  > recombination-equivalent density, and in the thin-layer phases the gap **is** the dust sink to
  > three decimals. Net: `P_HII/P_ram` median 6.165 → 1.545 on the profile form — still
  > HII-dominated on every row, by ~50% rather than ~500%.
  > **Batch 10 (2026-08-17) — the last cheap escape route died.** Batch 9 left a lead: extrapolating
  > stage 3's `Lw^−0.33` put inversion at a physical `Lw ≈ 3.4`. Tested on `B3MW3`/`B3MW10` and
  > **falsified** — the profile form does not inherit the cavity exponent, because stronger winds
  > thin the shell (`dR_ion/R2 ∝ Lw^−0.3375`) and that *raises* the geometry correction, cancelling
  > ~43% of the decline. Net response `Lw^−0.1133`, inversion at `Lw ≈ 46.5`, still unphysical.
  > G10.1/G10.3/G10.4 passed, **G10.2 failed** at 1.1902 against a registered < 1.0. Genuine
  > refinement: the geometry correction's **sign is wind-dependent**, which reconciles Batch 9's
  > scope ("raises") with its verdict ("lowers") — both were partial views of a thickness-dependent
  > sign. This is what left D5 load-bearing on measurement rather than on absence of alternatives.

- **2026-08-18 (maintainer re-sent the three docs; evaluation pass — K6's central identities
  independently verified, K7 lands at parity on fresh numbers)** — The maintainer uploaded
  `LITERATURE_ASSESSMENT.md`, `PLAN.md` and `README.md` with "save them into the relevant docs if
  need be; evaluate". **All three are byte-identical to HEAD (`c50aee70`; diff = 0 lines on each),**
  so there was nothing to fold in — and, notably, the *corrected* assessment the maintainer said was
  coming has not yet materialised as a distinct document; the ⛔ pending-correction banner stays
  until the maintainer either sends a revision or confirms the current text stands.
  **The evaluation's substantive output: the two claims K6 leans on are no longer author-verified
  only.** Under C-0.5 an external document is never load-bearing, and until today the assessment's
  §2.1/§4.2 identities — the C3c switch point being exactly Lancaster's `R_ch`, and the shipped
  branches being the CEM's exact asymptotes — rested on its author's own SymPy. Both are now
  re-derived in this workstream (`harness/cem_closure_check.py`, no trinity run, scale-free):
  crossover/`R_ch` = 1 to **4.3e-15** over 200 random draws spanning ~12 decades; the two `R_ch`
  forms agree to 2.6e-15; the asymptote ratios are 1.0007/1.0005/1.0000; and the crossover table
  reproduces Lancaster's to the digit — `F_sum/F_CEM` = **1.3421**, `F_max/F_CEM` = **0.6710** at
  `R_i = R_ch`, with every other cell matching. K6's tier moves S → **S + M**.
  ⚠️ One numerics lesson recorded because the first attempt produced a false failure at 2.4e4×:
  `brentq`'s default `xtol` is **absolute** (2e-12), and `R_ch` reaches 1e-17 in the draws, so the
  solver "converged" with the bracket still five decades wide — while the identity itself held to
  1.4e-16 at the same draw. Scale-free root-finds belong in log space. The committed harness says
  this in its own comments so the mistake is not re-made.
  **K7 arithmetic on our own numbers.** The assessment's α_p table used the stage-3-derived
  `P_C3a/P_ram` = 3.8–7.6; B11.0's fresh B3M momentum measurement is tighter, 5.091–7.156. At
  Paper II's `α_p` = 6.20 that gives **0.82–1.15 — straddling parity**: at the published
  calibration, B3M's momentum phase sits *at* the co-evolution crossover, neither wind- nor
  HII-dominated. That is the Lancaster picture arrived at from trinity's own measured range, and it
  is also why K6 and K7 are not competitors — `R_ch ∝ α_p²ṗ²/Q0`, so α_p is a *parameter of* the
  coupled closure, and inside K6 its effect is smooth where under the shipped C3c switch it would be
  a hard branch flip.
  **Where the value sits, stated for the record** (the maintainer asked): the single most valuable
  item in the influx is **K6** — it is the only candidate that supplies the second equation B11.A
  proved C3a is missing, it subsumes K5's four-source volume result, its asymptotes are the shipped
  branches, and its crossover is where the shipped switch already sits. The most valuable *next
  action* remains **B11.G on real trajectories** (cheap, offline, no source change), now de-risked
  by rung 0. K9 (+8.6–9.2% at B3M, literature precedent) is the best cheap orthogonal correction.
  Artifacts: `harness/cem_closure_check.py`, `data/b11g_cem_closure_check.csv`. No `trinity/`
  source touched; ship-hold unchanged.

- **2026-08-18 (revision 2 of the external assessment lands — α_p withdrawn by its author and
  closed by measurement here; the register re-ranks)** — The maintainer delivered the promised
  correction, twice over: a chapter-formatted `.tex` and a Markdown twin. The Markdown is committed
  verbatim as the new body of `LITERATURE_ASSESSMENT.md` (the `.tex` duplicates it, carries thesis
  styling and references a figure asset not supplied — not committed; say the word if it should be).
  The ⛔ do-not-act banner on revision 1 is **lifted**; rev1 is preserved in git history at
  `9aedeb45` and in rev2's own 22-item errata table, which names each rev1 claim that fell — the
  document retracts its own headline (`α_p` "wrong and is withdrawn in full"), reverses the Haid
  "tension", fixes the Geen-2022 chronology, and absorbs every B11/B12 correction this workstream
  had recorded against it. This is what an external input converging with the measurements looks
  like, and it is the outcome C-0 was designed to let happen without ever having been load-bearing.
  **Rev2's new claims were verified same-day rather than trusted** (C-0.5): `ENERGY_FLOOR` handover
  sites; `params['R1']` single-consumer; `run_transition_phase.py:331` reporting-only (delegation to
  `get_ODE_Edot_pure` confirmed at `:231`) — which also **corrected this plan's own D5 site count**,
  fixed in place, dated; the α_p convention identity (`eq:alphap_shock` ≡ `(R2/R1)² + (3/16)(R1/R2)²`
  at `R_f = (√3/2)R1`, to 1e-12 — rev1's "4/3 mismatch" was backwards); and the `f_abs⁻¹` outward
  crossover shift (rev1's `f^{1/3}` wrong in exponent and sign). One give-back: the 0.487 rev2
  removed as unsourceable is `ratio_min = 0.4873` in `data/b7_confinement_screen.csv`.
  **The decisive item: rev2's central open question was answered from committed data within the
  hour.** Rev2 asks whether TRINITY's `(R2/R1)²` — its α_p-equivalent, exact under the convention
  identity above — collapses abruptly to 1 at the 1c→2 handover (a genuine finding if so) or arrives
  there smoothly (in which case momentum is behaving exactly as Lancaster's theory says).
  Measured on `data/b7_regime_trajectory.csv`, both arms, `t > dt_switchon`:
  energy **8.4–12.2**, implicit **12.9–44.7**, transition **37.9 → 1.0018** — hitting 1 to **0.2%**
  exactly as `Eb` reaches `ENERGY_FLOOR` (c3c 1.0018, stock 1.0017). **No discontinuity. The α_p
  question is closed**: `α_p = 1` in the momentum phase is what the dynamics delivers, not an
  omission, and the momentum-phase HII dominance is entirely a `P_HII` question — the balance
  volume and closure — which is precisely where Batch 11 put it. K7 is struck from the register
  (dead by author-withdrawal AND by measurement); what survives of it is the diagnostics bundle
  (`(R2/R1)²`, `ζ`, `R2/R_ch`, `C_w`), folded into B11.G. Recipe recorded in the register row: the
  ratio of two committed columns, squared, excluding `t ≤ 1e-3`; no run, no new harness.
  **Where the register now stands after rev2**: K5 (balance volume) and K9 (shell-mass debit) are
  rev2's top two "by measured impact" — both already measured here (K5's momentum profile-form
  1.545; K9's +8.55/+9.22%) — with K6 as K5's principled completion ("K5 as the minimal move, K6 as
  the right one"). K7 dead. K2 (C3a-raw) unaffected but now competing against a register whose
  strongest rows all point at the volume/closure, not at branch logic. The evaluation of the
  previous visit stands with one amendment: rev2 does not dethrone K6, it *sequences* it —
  volume first (cheap, internal-consistency fix), closure when the ablation is affordable.
  **Open questions rev2 leaves and this visit did not settle**: whether `energy_phase_ODEs.py:385`
  is truly unreachable (rev2 pass-2 claim, unverified); whether `run_energy_implicit_phase.py:532`
  is live (assumed, unverified); and the under-dissipation reading of TRINITY's 8–45 vs Paper II's
  4.6–6.8 (needs careful θ bookkeeping; B11.G-adjacent). No `trinity/` source touched; ship-hold
  unchanged; hold-release criteria unchanged (met since B11.A–D; the release remains the
  maintainer's call, now with a cleaner register to rule on).

- **2026-08-27 (F_rad double-count checked — CLEAN; confinement geometry clarified; K10 registered
  with Batch 13 gates)** — Three outcomes of the maintainer's physics discussion, recorded before
  any measurement runs.
  **(1) F_rad is NOT double-counted, in either implementation.** Draine's trap — the hydrostatic
  ionised layer's *outer-edge* pressure already internalises the integrated radiation body force, so
  driving on that edge pressure AND adding `F_rad` counts it twice — was checked against source.
  Trinity's `get_shellODE.py` **is** Draine's system (`dndr ∝ n σ_d(L_n e^{−τ} + L_i φ) + …`), but
  that radiation-loaded profile feeds only the shell *structure* (f_abs, τ, gravity sampling). The
  drive never comes from it: old code drove on the capped `n_IF_Str` (≡ `Pb`, inner face), new code
  on the uniform-n Strömgren `P_C3a`. Inner-face surface pressure + `F_rad` as a body force is the
  *correct* control-volume decomposition. **Verdict: clean.** What the check exposed instead: in the
  driving branch, `P_C3a + P_ram` applied at `R2` is consistent bookkeeping **only if the ionised
  gas is interior to `R2`** (cavity gas, mass not in the shell). If it is the shell's own inner
  layer — which the four-source volume result says — then the shell's inner face sees only `P_ram`
  and the layer's push on the neutral part is an internal force. §6b seam C and the pressure
  composition are therefore **one defect seen from two sides** (mass ledger / force ledger), not
  two. Two small prints: `P_ext` is applied over `4πR2²` rather than the outer radius (matters only
  where the shell is thick — i.e. momentum), and `F_rad`'s ionised-layer absorption share is a
  legitimate body force.
  **(2) Confinement, stated precisely** (correcting a reading the maintainer floated, as asked):
  `R_i ≥ R_w` always — the cavity is transparent (B11.0: `φ = 1` across it), so the ionisation front
  cannot sit inside the wind bubble. "Confined" means `R_i − R_w ≪ R_w`: the wind fills essentially
  all the ionised volume and the photoionised gas is a thin dense skin on the shell's inner face.
  The shell **is** ionised — in that skin; neutral beyond the I-front; `f_esc = 0`. The photons are
  not "in equilibrium within the cavity" (nothing there absorbs); the budget is spent in the skin,
  by recombination **and dust** (61–75% measured). The clean statement of the two branches:
  **confined — the wind sets the skin's density, Strömgren balance sets its thickness; unconfined —
  Strömgren balance sets the density, the pressure follows.** Same two equations, opposite
  causality. A genuinely neutral shell exists only in the full Weaver-trapping limit.
  **(3) K10 registered** (§7.1) — the "better than exactly 0.0" the maintainer asked for: the
  coupled closure in trinity variables, returning the smooth excess
  `P_conf·[(R_i/R2)² − 1]` with `R_i` from recombination over the cavity-excluded layer at the
  pressure-equilibrium density. Exact first-order confined limit (Lancaster's own
  `(2/3)(R_w/R_ch)` term), kills the switch discontinuity and the §3c.1 kink by construction,
  contains K5's volume fix, needs no root-find because trinity's `R2` **is** `R_w`. Batch 13's
  gates (G13.1–G13.5) are registered above **before** the screen runs, including the disclosed
  expectation that the no-dust variant will overshoot the shipped momentum drive and the
  pre-registered rule that a >2× dust sensitivity reads "cannot ship without a dust model".
  No `trinity/` source touched; ship-hold unchanged.

- **2026-08-27 (Batch 13 — K10 screened; two of my own gates failed, and the dust rule fired)** —
  Ran the K10 offline screen against gates committed at `93069d28`, before the harness existed.
  **Headline: the thing K10 was proposed for works, and the thing that blocks it is dust.**
  The shipped rule's drive jumps **+34% at fixed state** when the branch flips (B3M 1.3494e3 →
  1.8079e3; B3MW01 +33.2%); K10's jump is **identically zero**, being a single-valued smooth
  function with no branch. And on B3M's energy+implicit rows — the branch §6b found exactly
  consistent — K10 moves the drive by a median **+0.68%**, so it does not re-decide a healthy phase.
  ⛔ **But G13.4's pre-registered dust rule fired at 2.05×**, and the verdict stands as written:
  **K10 cannot ship without a dust model.** The no-dust form gives 9.0–14.0 `P_ram` on B3M momentum
  against the shipped 6.1–8.2 — my **disclosed expectation of 9.5–14.7 was right**, which is the
  value of writing it down first. With `Q_eff = Qi f_abs (1 − f_dust,ion)` it falls to 4.9–8.5,
  i.e. **0.851× the shipped drive**, and its predicted `R_i/R2` = 2.39 lands on the shell solve's
  measured `R_IF/R2` ≈ 1.7–2.3 where the no-dust 3.39 does not. **The substantive finding is a
  cancellation**: C3a's cavity-volume inflation and its missing dust sink are similar in size and
  opposite in sign here, so C3c is closer to the coupled answer than its individual defects imply —
  for compensating reasons, which is precisely the kind of agreement not to extrapolate.
  ⚠️ **Two of five gates failed, both my fault, both recorded as failed rather than re-barred.**
  **G13.1** measured the drive's *row-to-row step* at a switch; adjacent snapshots contain real
  evolution, so the metric conflated a jump with change — the shipped rule scores 5.97–6.79% by the
  same metric, which is the tell. Re-measured at fixed state it is +34% vs 0%. The registered metric
  was wrong; the property holds by construction. **G13.3** found the two K10 forms disagreeing by
  6.53e-2, which is exactly `chi_e^{2/3} − 1` = 0.0656: Lancaster's `eq:ionreceq2` carries **no**
  electron factor while trinity writes `chi_e·alpha_B·n_H^2`, so **`R_ch`(trinity) = `chi_e` ·
  `R_ch`(Lancaster)**. That propagates backwards into the same-day verdict entry, whose CEM
  comparison omitted `chi_e`: `new/CEM` 0.548–0.638 → **0.583–0.679**, `old/CEM` 0.134–0.210 →
  **0.143–0.223**. **No conclusion changes** (old stays 3–7× low under every mapping) and the
  figures are corrected in place; `harness/cem_closure_check.py` is unaffected, being explicitly
  scale-free with `chi_e` folded into `alpha_B`.
  **Scope, stated so it is not over-read**: an offline screen on committed trajectories — every row
  uses the shipped run's `R2(t)`, so it cannot say what K10 would do to a trajectory it drove
  itself. No fate, no ΔR2, no `trinity/` change, ship-hold intact. Next rung for K10 would be a
  dust-carrying variant plus a shadow-arm run through the full equivalence ladder — **not started,
  and it needs the maintainer's ruling on the register first.**
  Artifacts: `harness/k10_cem_drive_screen.py`, `data/b13_k10_screen.csv`.

- **2026-08-27 (K5 and K9 gated as Batches 14/15 — and K9's pre-gate demoted it before it ran)** —
  Maintainer asked for gates on the two candidates rev2 ranks highest. Drafted, and the drafting
  itself produced a correction I had to make to my own advice from an hour earlier.
  **K5 (Batch 14)** is the clean one: swap `get_phii_c3c`'s denominator from the wind cavity
  `(4/3)πR2³` to the cavity-excluded layer that `shell_structure.py:243` already uses. Two variants
  registered (analytic layer / profile), with **G14.0 blocking**: regress `P_HII` on `Pb` and fail if
  the slope lands in [0.95, 1.05] with r² > 0.99. That gate exists because K5b reads a profile whose
  *inner boundary* is `nShell0 ∝ Pb` — the original `P_HII ≡ P_conf` circularity could walk straight
  back in through the fix for it, and nothing in the register had noticed. Conventions (`χ_e`,
  `R_IF`, exact spherical volumes) are pinned in the gate text, because a convention mismatch is
  exactly how G13.3 failed today.
  ⚠️ **K9 (Batch 15) is demoted, by its own pre-gate, before any implementation.** I had told the
  maintainer K9 was "measured at +8.55/+9.22%, a cheap independent win". **That figure debited
  `M_cav` — the mass C3a's *cavity* premise implies — which is not the K9 quantity.** In trinity's
  geometry the ionised gas is the shell's own inner layer, so K9 debits the *layer* mass. Measured
  it (integrating the shipped profile, B3M): `m_ion/m_shell` median **0.461** in momentum, range
  **0.109–1.000**, and **1.0000 throughout the energy phase**. Three things follow, none of which
  were visible before: the **mass** fraction is nothing like the 0.9954 **thickness** fraction
  (the profile rises steeply outward, so the mass sits in the thin neutral skin); it **reaches 1.0
  on real rows**, i.e. a massless shell and a divergent `vd = F/M`; and the profile integral only
  reproduces `shell_mass` in transition/momentum (1.000–1.002), so energy/implicit cannot even pose
  the question yet. G15.2 makes the floor blocking with the pre-registered expectation that energy
  and implicit **fail** it. And G15.3 raises the question K9 cannot dodge: if the layer is debited
  from the shell it is interior gas transmitting pressure, so the drive should act at `R_IF` on the
  neutral shell — debiting the mass while still driving at `R2` swaps today's inconsistency for the
  opposite one. That may make K9 **inseparable from K5**, which is the same
  one-defect-seen-from-two-sides pattern the `F_rad` check turned up.
  **Ordering revised on the evidence: K5 first, alone.** Both batches carry a full-run equivalence
  gate (CLAUDE.md rule 5 — separate processes, matched `t`, stiff regimes) and are to be run
  **separately**, since K5 lowers the drive and K9 lowers the inertia and a combined arm could show
  a null while hiding two real changes. No `trinity/` source touched; ship-hold unchanged.

- **2026-08-27 (portability audit — three real defects in my own record, fixed)** — Maintainer asked
  whether this workstream is portable to a fresh session, whether the source-of-truth rule still
  holds, and whether anything is stale. Audited rather than asserted; found three defects, all mine.
  **(1) Dated-log entries were misdated by nine days.** The last three log entries and the Batch
  13/14/15 sections were written on **2026-08-27** but stamped 2026-08-18 (the session began on the
  18th and I carried the date forward without re-checking). §0's step 3 calls an undated change
  "contamination of the record"; a *mis*dated one is worse, because it silently reorders causality —
  a reader would have concluded B13's result predated Batch 12's, which it does not. Corrected in
  place: entries and section headers for the F_rad/K10 registration, Batch 13, and Batches 14/15 now
  read 2026-08-27, matching their commits (`93069d28`, `caf38d58`, `d234e945`). Everything at or
  before `30b01fd3` genuinely is 2026-08-18 and is unchanged.
  **(2) G15.0's numbers had no harness — a 💾 violation.** The K9 ionised-mass-fraction measurement
  that demoted K9 was run as a throwaway inline script; the numbers were quoted in the gate text
  with **no way for a later session to reproduce or extend them**. That is exactly the failure mode
  the 💾 banner exists to prevent, and it was in the very batch that turns on those numbers. Fixed:
  `harness/ionised_mass_fraction.py` + `data/b15_ionised_mass_fraction.csv`, with a reproduce
  command in the gate. Writing it properly also **added a config and changed the finding**: on
  `B3MW01` the momentum median is **0.1494**, against B3M's 0.4611, so the ionised mass fraction is
  **strongly regime-dependent** and K9's magnitude cannot be quoted from one config. Across both
  runs **31 of 72 rows** sit at `m_ion/m_prof ≥ 0.95` — in energy, implicit **and momentum** — so
  G15.2's admissibility problem is broader than the single-config measurement suggested. The harness
  also emits `profile_trustworthy` (`m_profile/shell_mass ∈ [0.98, 1.02]`), because the profile
  integral only accounts for the run's shell mass in transition/momentum; in energy/implicit it is
  up to 2× off and the fraction there **must not be used**.
  **(3) `DOC_STATUS.md` and `README.md` had gone stale** — neither mentioned Batch 13, 14, 15 or K10.
  Both reconciled. The source-of-truth rule is unchanged and still holds: **`PLAN.md` is the one
  doc**; `README.md` is the pre-C3c evidence record; `LITERATURE_ASSESSMENT.md` is an external input
  under the C-0 carve-out and is **never load-bearing**; `DOC_STATUS.md` carries one row per
  workstream. No `trinity/` source touched; ship-hold unchanged.

- **2026-08-28 (Batch 14 offline screen — the K5 premise inverts under its own pre-gate's
  diagnostics)** — Maintainer asked to look at the φ_II picture, what a proper implementation
  of the K5 volume fix looks like, and what would change; ship-hold unchanged, so this visit is
  offline-only. Built `harness/k5_offline_screen.py` (committed before first run) over committed
  CSVs alone: B3M via the `b9_layer_density` × `b11_mass_ledger` `row_idx` join (join integrity
  re-verified at ≤1.5e-7; route check: reconstructed `P_C3a` = stored `P_HII` to ≤3.5e-7 on all
  33 driving rows), B3MW01 via the b12 photon ledger (27 driving rows, K5a only — no committed
  `n_rms` there, recorded as a coverage cap). Findings, in order of weight:
  **(1) G14.0 does not trip as barred, and the bar itself is the lesson.** No variant lands a
  linear slope in [0.95, 1.05] with r² > 0.99. But the log-log diagnostic disclosed alongside
  shows `P_K5x ∝ Pb^{1.0}` on the driving rows (K5a +1.016/r² 0.988, K5b +1.020/r² 0.993) at
  gain ≈2.4/≈1.5 — the original identity with an O(1) gain, which a linear-slope bar centred on
  gain 1 cannot see. Recorded pass-by-the-letter, not re-barred; the finding is flagged instead.
  **(2) K5a is not a new quantity.** It is algebraically the uncapped `n_IF_Str_raw` the code
  already computes (`shell_structure.py:247-251`, same balance, same `_vol_ion`, modulo
  `f_abs` vs `(1−f_esc_ion)` bookkeeping) — so its coupling was already measured: Batch 3b,
  788 rows, `slope_nIFraw_vs_n0` 1.005–1.096 at r ≥ 0.9971; and its full-run cost class too:
  Batch 4a, ΔR2 15.3–28.4%, breaches concentrated in the `dt_switchon` window. The workstream
  measured K5 before K5 had a name, and §3b already said why: `shell_n0 ∝ Pb` is the shell
  ODE's inner boundary, so any balance over the shell's own layer inherits `Pb`.
  **(3) The branch census makes the swap non-minimal.** 83/83 confined B3M rows flip to
  driving under both variants (energy medians: K5a 2.248×`Pb`, K5b 1.073×`Pb` — the identity
  to 7%), first flip at t = 8.3e-7 inside the ramp window: a bare denominator swap re-admits
  the D-ramp class C3c closed, in every phase.
  **(4) What would change if K5 were scoped to the driving branch only:** momentum
  `P_HII/P_ram` median 6.165 → 2.381 (K5a) / 1.545 (K5b) on B3M — both priors from the gate
  text hit exactly — and 13.766 → 4.388 (K5a) on B3MW01; transition 3.952 → 2.559/1.587. No
  driving row flips confined on either config, so the momentum phase stays HII-dominated
  (D5's magnitude question survives K5 at ~1.5–2.4× rather than ~6×).
  **Net: the premise ran backwards.** `get_phii_c3c` vs `n_IF_Str` inconsistency is real, but
  the cavity volume is the *decoupled* side and the layer volume the *coupled* one; the
  literature's layer volume lives inside explicitly coupled closures (Geen/Lancaster — K10/K6),
  where Batch 13 already measured the closure landing within 10–15% of shipped once dust is
  in. Batch 14's §8.1 row records this; the arm decision (registered bare swap for the record /
  driving-only rescope / defer the volume to K10) is D5-adjacent and the maintainer's.
  Siblings reconciled (README.md, DOC_STATUS.md). No `trinity/` source touched; ship-hold
  unchanged. Artifacts: `harness/k5_offline_screen.py`, `data/b14_k5_screen.csv`.

- **2026-08-28 (maintainer question: "if `P_HII == P_conf`, are we just double counting?" — measured
  per phase, and the answer splits three ways)** — Asked whether the identity holds in
  energy/implicit/momentum/transition and whether that makes the system a double-count. Measured
  rather than recalled: `harness/identity_census.py` → `data/b14_identity_census.csv`, over the
  committed both-arm trajectories (B3M `b7_regime_trajectory.csv`, B3MW01
  `b12_lowwind_trajectory.csv`). The key structural point, verified at source (`cce8c924`) and then
  **checked against the runs' own stored `P_drive` rather than asserted**: a pressure double-count
  needs BOTH the identity AND an additive composition, and the composition differs by phase —
  `max(Pb_eff, P_HII)` in energy/implicit (`energy_phase_ODEs.py:256`), `max(Pb_eff, P_HII + P_ram)`
  in transition (`:253`), `P_HII + P_ram` in momentum (`run_momentum_phase.py:445`). Recomputing the
  drive from stored components reproduces the stored `P_drive` **bit-exactly (0.00e+00)** in
  implicit/transition/momentum on both arms and both configs.
  **(1) Pre-C3c the identity WAS universal:** `frac_identity` = **1.0000** in implicit, transition
  and momentum on both configs (relΔ ≤3.6e-16 per Batch 0), 0.9855–0.9885 in energy (the documented
  stale-`Pb` 1a→1b handoff rows).
  **(2) But the double-count was NOT uniform, because the max phases absorb it.**
  `max(Pb, Pb)` = `Pb`, so energy/implicit drove at `P_drive/Pb` = **1.000** — no double-count there
  ever. Transition drove at **1.824** (B3M) / **1.741** (B3MW01) — a phantom `Pb` added to `P_ram`.
  Momentum drove at **exactly 2.000** on both configs, with `frac_Pb_eq_Pram` = **1.0000**
  confirming the momentum convention (`Pb` := `P_ram`) from the data instead of assuming it. So the
  literal "drive on twice the ram pressure" was real, and it was **momentum + transition only**.
  **(3) In production today the identity is GONE:** `frac_identity` = **0.0000** in every phase of
  both configs. `P_HII` is either exactly 0.0 (energy 100%, implicit 100%, transition 23.8%/9.4%)
  or the decoupled `P_C3a`, whose value contains no `Pb` (§6b verified item 3). **The
  pressure-relabelling double-count C3c was built to remove is measured removed.**
  ⚠️ **What is NOT resolved, and is what SHIP-HOLD is actually about:** the *additive* composition
  survives, and `P_HII` is now much larger than what it is added to — momentum `P_drive/P_ram` is
  **7.165** (B3M) / **14.766** (B3MW01) against the old code's 2.000. Whether adding a decoupled
  `P_HII` to `P_ram` is right is D1-vs-physics (K3 priced removing the `+P_ram` at −14.1%) and is
  open. And two **non-pressure** double-counts are CONFIRMED live: §6b seam A (photons — shell
  starts `phi0 = 1` while C3a spends `Qi·f_abs` on cavity recombinations, `f_abs` = 1.0000 on 29/33
  driving rows ⇒ budget ≈`2·Qi`) and seam C (mass — `shell_mass` already holds 100.0000% of the
  run's gas, so `M_cav` has no source; `(M_cav+M_shell)/M_avail` = 1.5638). Those are the standing
  reasons not to ship, and neither is fixed by any K-row that only changes `P_HII`'s value.
  🔗 **Consequence for Batch 14, worth stating plainly:** momentum composes additively, so a K5
  variant that is ∝`Pb^1.0` with gain ≈1.5 (K5b, this batch's finding) would drive at
  ≈`(1.545 + 1)·P_ram` = **~2.5×`P_ram`** — numerically near the old code's 2.000, and with a
  `Pb`-proportional term back in the sum. That is the pre-C3c *structure* returning at a different
  constant, which is the sharpest single argument against the bare swap.
  **Method note (a failed check, recorded):** the same recompute is **VOID in the energy phase** —
  the live drive there uses the *ramped* `press_bubble`, which is not a stored column, while the
  census uses the stored un-ramped `Pb`. On stock the two coincide (`P_HII ≡ Pb` won the max, err
  2.2e-16); on c3c the drive is the ramp alone at **0.817/0.708 × stored `Pb`** and the recompute
  overstates by up to 2.03×. That gap **is** the D-ramp mechanism (§3 item 3) reproducing
  independently, not a composition error; the harness marks the energy rows
  `recompute_check=void` rather than reporting a spurious failure. No `trinity/` source touched;
  ship-hold unchanged.

- **2026-08-28 (maintainer question: what is the physically correct next step, and does K10 carry
  Lancaster's discontinuity fix? — plus a composition defect found while answering)** — Two
  questions, one answer: the discontinuity fix and the seam fix are the same construction.
  **(1) Where the jump actually is.** §3c.1's "the `max` is C0, only the derivative kinks" is
  correct *for the max phases* and does not generalise — and the reason is the same sum-vs-max
  split the identity census measured today. In `max(Pb, P_HII)` the branches are equal at the
  crossover, so nothing jumps. In the **summing** compositions the switch is a genuine
  discontinuity in value, because `P_HII` enters additively: Batch 13 measured it at fixed state
  as **+34.0%** (B3M, transition, t = 0.3012) and **+33.2%** (B3MW01, t = 0.7186). So the
  discontinuity is a property of `sum` + `branch`, not of `max`.
  **(2) Yes, K10 is the Lancaster construction, and its smoothness is structural.** `P_drive =
  P_conf·(R_i/R2)²` is a single-valued smooth function of `(R2, Qi_eff, P_conf)` with **no branch**,
  algebraically Lancaster's `α_p ṗ (1 + R_w/R_ch)^{2/3}` at `α_p = 1` in the MD phase; its measured
  state-jump is **identically 0.0%** against the shipped rule's +34%. It interpolates between the
  wind- and photoionisation-dominated limits instead of selecting between them, which is why it also
  removes the §3c.1 `t_cross` kink and the transition-entry step.
  **(3) The strategic argument, stated as evidence.** B11.A's degeneracy result (`x* = 1` on 33/33
  driving rows) says seams A and C **cannot be repaired inside C3a's structure at all** — so every
  candidate that computes an independent `P_HII` on top of an unmodified shell solve (K1, K2, K4,
  **K5**) inherits the photon and mass double-counts regardless of what value it computes. Batch
  14 sharpened this from the other side: the layer volume the literature uses is *coupled*, and it
  only stops being a relabelling when the coupling is structural — which is exactly what K10/K6
  are. K10 also contains K5's volume fix by construction, and its dust-corrected `R_i/R2` = 2.39
  lands on the shell solve's own measured `R_IF/R2` ≈ 1.7–2.3 where the no-dust form (3.39) does
  not. **The coupled closure is where the evidence points; D5 remains the maintainer's ruling.**
  **(4) ⛔ A composition defect in K10's own specification, found while answering and corrected in
  the §7.1 row.** "One helper, zero `P_drive` edits — the excess rides the existing compositions"
  is true **in momentum only**. Momentum (`P_HII + P_ram`) composes the excess exactly;
  energy/implicit (`max(Pb, P_HII)`) needs the **full value** instead and **swallows the excess
  whenever `(R_i/R2)² < 2`**; transition (`max(Pb, P_HII + P_ram)`) is exact under neither. That
  discards K10's advertised confined-branch improvement — the first-order term
  `(2/3)(R2/R_ch)·P_conf` is small by construction, so `max(Pb, small)` throws it away precisely
  where it was supposed to beat the 0.0. Batch 13 could not have caught this: its screen computed
  the total drive directly rather than routing a helper return through each phase's `P_drive`
  expression. **Consequence: K10 needs a phase-aware helper or real edits at the live `P_drive`
  sites, so it is not a one-helper zero-edit change and its risk class rises.** This is algebra,
  not a measurement — it needs its own pre-registered gate before any K10 work, and that gate
  should compose through the real expressions, not around them.
  **Recommended ordering (evidence-based; not a decision):** (a) rule D5 on the *family* question —
  independent-value (K1/K2/K4/K5) vs coupled-closure (K6/K10); (b) if coupled, fix the composition
  mapping above and pre-register it; (c) build the dust term **inside** the closure rather than
  joining `f_dust` post-hoc from the photon ledgers as Batch 13 did — G13.4's 2.05× is the blocking
  gap and G9.4 already measured the sink at 61–75% of the absorbed budget; (d) only then an arm,
  on the Batch 14 ladder. No `trinity/` source touched; ship-hold unchanged.

- **2026-08-28 (D5 reframed as one physical question — and a correction to my own claim from
  earlier today)** — Maintainer: "I honestly don't know what to do with D5." Re-read D5 as posed
  rather than as accreted, and it is stuck for a structural reason: **the cell now bundles three
  different questions**, one of which is already answered by measurement and one of which is not a
  decision at all. Decomposed:
  **(A) Is the cavity premise right?** Not a preference — measured, three ways, and the answer is
  no: seam A (photons spent twice, `f_abs` = 1.0000 on 29/33 driving rows ⇒ budget ≈`2·Qi`),
  seam C (`shell_mass` already holds 100.0000% of the run's gas, so `M_cav` has no source), and
  B11.A (`x* = 1` on 33/33 — C3a cannot be closed photon-conservingly at all). K1/K2 rest on it.
  **(B) What does the ionised layer transmit? — this is the real D5, and the code currently answers
  it BOTH WAYS at once.** `shell_structure.py:125` sets the shell ODE's inner boundary to
  `nShell0 = (μ_i/μ_c)·Pb/(k_B T)`, which *is* the statement "the ionised skin is in pressure
  equilibrium with the bubble" (§6b item 5 says so). A layer in equilibrium transmits `P_conf` and
  adds nothing — C3c's confined branch. But on the driving branch the drive asserts a layer density
  **1.30–7.16× (median 5.09×)** higher than that same boundary condition
  (`b11_photon_ledger.csv:shell_n0_ratio`). **So the code assumes equilibrium in the shell solve and
  non-equilibrium in the drive, for the same gas at the same instant.** That is seam B, and it is
  the one thing every candidate must resolve. Three coherent resolutions, and only three:
  **(i) equilibrium everywhere** — trust the shell solve; `P_HII` adds nothing ever; drive is
  `Pb`(+`P_ram`). Self-consistent, but predicts no photoionisation feedback at all.
  **(ii) non-equilibrium everywhere** — then `shell_structure.py:125` is wrong too and must change;
  the honest version of "`P_HII` is real", and a much larger change than any §7.1 row admits.
  **(iii) coupled/interpolating (K6/K10)** — equilibrium holds *at the inner boundary*, the
  ionisation front sits further out, and the drive is the equilibrium pressure amplified by
  geometry, `P_conf·(R_i/R2)²`. The independence moves from the *density* into the *geometry*, which
  is why it escapes: pressure-slaving becomes physics rather than relabelling, and `Qi` enters
  through `R_i`. Recovers both limits smoothly, no branch, no contradiction.
  **(C) Which register row to adopt** — downstream of (B), not a separate decision. Once (B) is
  ruled, at most two rows survive.
  ⛔ **Correction to my own framing from earlier today.** I wrote that K1/K2/K4/**K5** all "inherit
  the photon and mass double-counts no matter what value they compute". **That is too strong for
  K5** and I retract it. K5 balances over the *layer* — the same region the shell solve already
  owns — so it does not posit a second photon sink or a second mass reservoir, and seams A and C
  should largely dissolve under it. What Batch 14 actually measured is a **trade, not a strict
  loss**: K5 buys geometric consistency and pays for it by becoming `Pb`-slaved (∝`Pb^1.0`, r²
  0.988/0.993), because the layer's density is set by the `Pb` boundary condition. (That K5 relieves
  seams A/C is a *derivation from the geometry*, not a measurement — it needs its own gate before
  anyone leans on it.)
  **The sharp form of the dilemma, which is what makes D5 hard and is worth stating once:**
  within an independent-pressure architecture **decoupling and geometric consistency are in direct
  conflict**. Any balance over the shell's own ionised layer inherits `Pb` (the boundary condition);
  the only way to get a `Pb`-free `P_HII` is to balance over the cavity, whose ionised gas does not
  exist. D2's 2026-08-12 instruction ("`P_HII` should be a real, separate pressure") therefore asked
  for something this architecture cannot deliver consistently — which is not a failure of D2, it is
  what the last six batches discovered. Resolution (iii) dissolves the conflict rather than picking
  a side. No `trinity/` source touched; ship-hold unchanged; D5 remains the maintainer's ruling.

- **2026-08-28 (maintainer challenge to seam C's wording — upheld; and the first PRIMARY-source
  check of the closure literature, which changes the evidence tier)** — Maintainer: *"the shell
  should not hold 100%, because the bubble cavity has density and also has mass `mBubble` … I think
  it's just that we assume the gas in the cavity is invisible to ionising photons because they are
  already all excited due to T~1e7."* **Both halves land, and the second one is sharper than what
  the record said.** Measured: `harness/cavity_gas_check.py` → `data/b14_cavity_gas.csv`.
  ⛔ **Seam C's "`M_cav` has no source" is CORRECTED as loose.** The run does carry cavity gas —
  `bubble_mass` = **99.643 Msun** (B3M) / **31.639 Msun** (B3MW01), frozen across driving rows as
  B11.0 found. It is ~0.1% of `M_avail`, which is why `shell_mass/M_avail` rounds to 1.0000; the
  shell holding "100%" is a rounding statement, not a claim that the cavity is empty. The accurate
  statement is a **density overstatement of 71.9–576× (B3M, median 95.5) and 215.8–409.8 (B3MW01)**.
  ✅ **And the maintainer's physical reading is right, with a stronger consequence than "invisible".**
  Recombination goes as `n²α_B(T)V`, so over the same cavity volume the sink C3a *assumes* is
  available only to the fraction `(n_actual/n_implied)²` = **3.0e-06–1.9e-04 (B3M)** and
  **6.0e-06–2.2e-05 (B3MW01)** — **4–5 orders of magnitude, on the `n²` term ALONE, with no
  temperature model at all.** Temperature only deepens it. So **seam A is not "two models split one
  photon budget"; C3a posits a recombination sink that is physically unavailable**, and
  `shell_structure.py:120`'s `phi0 = 1` is *right* — which is what §6b item 5 already argued for the
  confined branch, now quantified. ⚠️ The bubble temperature implied by the run's own `Pb` and
  `bubble_mass` is **1.8e5–1.3e6 K**, below the ~1e7 the maintainer assumed; but `bubble_mass` is
  the frozen/unusable column, and if it is overstated then `n_actual` falls, `T` rises toward 1e7
  **and the sink ratio shrinks further** — the verdict is robust to that column in both directions.
  📚 **Primary-source check (first in this workstream; C-0 §5 bars `LITERATURE_ASSESSMENT.md` from
  being load-bearing, so these were read directly).** **Rahner, Pellegrini, Glover & Klessen 2017**,
  *"Winds and radiation in unison"*, MNRAS **470**, 4453, arXiv:**1704.04240**,
  doi:10.1093/mnras/stx1532 — WARPFIELD, trinity's direct ancestor; the shell is modelled as
  **quasi-hydrostatic**, i.e. trinity's `nShell0 ∝ Pb` inner boundary is that lineage's standard
  assumption and is **not** an error. **Lancaster et al. 2025**, *"The Co-Evolution of Stellar
  Wind-blown Bubbles and Photoionized Gas I"*, arXiv:**2505.22730**, ApJ **989**,
  doi:10.3847/1538-4357/ade66b (Paper II: arXiv:**2505.22733**) — the CEM source. Verified from the
  paper itself: it **explicitly assumes pressure equilibrium** ("the WBB has come into equilibrium
  with the PIR with no net force being applied across its outer edges"); its recombination balance
  `(4π/3)(R_i³ − R_w³) α_B n_H,i² = Q₀` is over the **cavity-excluded layer** (K5's premise, inside
  the coupled closure); and — the part no §7.1 row states — **the two forces are applied at two
  DIFFERENT radii**: the wind at `R_w`, the photoionized gas at `R_i`, a distinction the paper calls
  central to its framework. ⚠️ Author lists here are from the preprint landing pages and are **not
  yet ADS-verified**; verify before any `.bib`.
  🔑 **The synthesis this forces, and it reframes §7.1 again.** Trinity collapses `R_w` and `R_i`
  onto the **single radius `R2`**. That one fact generates both defects the workstream has been
  chasing separately: with one radius the wind pressure and the photoionized pressure are applied at
  the same place, so composing them *additively* double-counts (the `P_HII + P_ram` seam); and the
  ionised gas must simultaneously satisfy the equilibrium boundary condition at `R2` and supply a
  drive exceeding it, which is seam B's 1.30–7.16× contradiction. **Equilibrium is not the problem —
  one radius doing two jobs is.** Lancaster keeps equilibrium *and* obtains a driving pressure by
  separating the radii and letting `(R_i/R_w)` carry the amplification. This raises **K8
  (three-radius)** from tier **A** (assessment-only, "pending") to **S** — it is the primary
  source's own structure, not a speculative row — and it means K6/K10 are the *reduced* forms of
  K8, not alternatives to it. No `trinity/` source touched; ship-hold unchanged.

- **2026-08-28 (maintainer ruling: KEEP ONE RADIUS — and Batch 16 closes the composition question
  the same day)** — Maintainer, on the two-radius question raised by the Lancaster+2025
  primary-source check: *"so lets just keep one radius now."* Recorded as a scope ruling, and it
  settles more than it looks:
  **(a) K8 is deferred**, not killed — the primary source's two-radius structure stands as the
  physically fuller picture (tier **S** as of today) and remains the natural follow-up paper, but it
  is out of scope for this workstream.
  **(b) `shell_structure.py` is NOT touched.** Its quasi-hydrostatic inner boundary is Rahner+2017's
  own modelling assumption (arXiv:1704.04240) and Lancaster+2025 assumes pressure equilibrium too,
  so the shell solver is the part of this system behaving correctly. A shell-solver workstream was
  considered and **declined on the evidence** — sizing, for the record: `shell_structure.py` 478
  lines + `get_shellODE.py` 153, 5 phase-runner callers, ~34 attribute reads, ~10 test files. Cheap
  as a module, but a two-radius change would move `R_w` into tracked state, relocate the force sites
  in four runners and move essentially every trajectory golden — paper-scale, which is what K8's own
  row said.
  **(c) K10 is the live candidate**, being the one-radius reduction of Lancaster: `R_i` is computed
  algebraically and never tracked.
  **Batch 16 registered and run the same day, gates committed first.** The composition defect Batch
  14 found is **solved**: one rule — `return = P_conf·ρ − (P_ram if this phase's composition adds
  it)` — reproduces the CEM drive through **all three real `P_drive` expressions** to **2.22e-16**
  (G16.0, both `Q_eff` variants), every return is non-negative (G16.1, 853/853), and Lancaster's
  confined-branch first-order term is now **delivered at +0.96% over `P_conf`** rather than being
  swallowed by the `max` (G16.2) — the precise failure Batch 14 identified. G16.4 reproduced Batch
  13's magnitudes exactly through a different code path, which is the strongest cross-check either
  batch has.
  ⚠️ **One thing got harder, as pre-registered.** G16.3: inside `dt_switchon` the ramped/un-ramped
  `P_conf` ratio is **0.3302–0.9952 (median 0.7112)**, so K10 must receive the **ramped**
  `press_bubble` — and `params` carries `current_phase` but not `press_bubble`. **K10 therefore
  needs a signature change** (`get_phii_c3c(params, shell_props, P_conf=None)`), small and additive
  but still a `trinity/` edit, so still behind the ship-hold.
  **Remaining K10 blockers are now exactly two: the dust model (Batch 17 — G13.4 fired at 2.05×)
  and a full-run arm.** No `trinity/` source touched; ship-hold unchanged; D5's family question is
  effectively resolved toward the coupled closure by this ruling, and should be marked as such in
  §7 when the maintainer confirms that reading.

- **2026-08-28 (Batch 17 — dust put inside the closure; G13.4's blocker is discharged, and one of
  my own pre-registered expectations missed)** — Maintainer: "do batch 17". Gates registered and
  committed before any measurement, per §0. The design choice that made this cheap: **do not invent
  a dust model.** `get_shellODE.py:120` already carries the code's ionised-region photon equation
  with both sinks (`dφ/dr = −4πr²χ_e α_B n²/Qi − n σ_d φ`), so K10's dusty closure is that same ODE
  integrated at the closure's uniform `n₀` — a *reduction* of trinity's own treatment, which is also
  what makes G17.0 a meaningful validation rather than a self-consistency tautology. `Qi` is used
  whole, not `Qi·f_abs`, because the shell solve starts at `φ = 1` and the
  recombination/dust/escape split is an output.
  **The headline: the closure's dust fraction reproduces the shell solve's own to 5.6%** (median
  predicted/measured **1.056**, 97.3% of rows within 25%, B3M and B3MW01 agreeing independently at
  1.064 and 1.052). I had disclosed this gate as genuinely at risk, because G9.4 measured the
  uniform analytic form overstating the profile's recombination-equivalent density by up to 3.17×.
  It passed, and the reason is structural rather than lucky: **dust absorption is linear in `n`
  (`∫ n σ_d φ dr`) where recombination is quadratic**, so the dust fraction is far less sensitive to
  profile shape than the quantity G9.4 was measuring. The uniform reduction is a poor density model
  and a good dust model, and that asymmetry is exactly what K10 needs.
  ⛔ **G17.3's pre-registered expectation MISSED and is recorded as such.** I predicted the
  self-consistent drive would land *between* the no-dust and post-hoc values, nearer the post-hoc.
  It lands just **below both** on 4 of 5 phase×config groups (`c/b` = 0.92–0.95). Direction right,
  containment wrong. Diagnosis: the post-hoc form debits `(1 − f_dust)` once up front, while the
  closure removes photons continuously along the layer where dust competes with recombination at
  every radius, so it limits `R_i` slightly more. Re-running G13.4's sensitivity with the closure in
  place of the join still gives **1.886–2.214**, above the old 2× bar on B3MW01 — but that bar was
  never "make dust matter less"; G13.4's verdict was *"K10 cannot ship without a dust model"*, and
  the deliverable was a model. Dust is now computed inside the closure and validated against the
  code's own solve, so it is no longer a free knob. **Blocker discharged.**
  **End-to-end, the complete candidate** (dust closure + Batch 16's mapping, composed through the
  real `P_drive` expressions) sits within ~25% of the shipped drive everywhere and within ~10% on
  B3MW01, with B3M's healthy branch essentially untouched (1.005–1.006). Layer dust optical depth is
  0.246–4.521 (median 1.591), so the layer is marginally optically thick to LyC — consistent with
  dust mattering at ~2×.
  ⚠️ **One honest coverage gap, flagged rather than buried:** G17.0's comparison rows are transition
  70 / momentum 42 / implicit **1**, because the photon ledgers only ever replayed the driving
  branch. **ED-phase dust is validated on a single row**, so G17.4's energy/implicit columns rest on
  an unvalidated dust fraction there. Extending `harness/photon_ledger.py` to confined rows is cheap
  and is the obvious first step of any follow-up.
  **Where K10 now stands: both offline blockers are cleared.** Batch 16 solved the composition,
  Batch 17 the dust. What remains is (i) the `get_phii_c3c(params, shell_props, P_conf=None)`
  signature change G16.3 established as mandatory, and (ii) a **full-run arm** — both `trinity/`
  work, both behind the ship-hold, and the arm ladder is already staged in
  `docs/dev/phii-identity/hpc/b14/`. Nothing further can be settled offline. No `trinity/` source
  touched; ship-hold unchanged.

- **2026-08-28 (ship-hold lifted for the K10 arm only; K10 implemented and per-call gated — and my
  own BLOCKING gate failed, which is the useful part)** — Maintainer lifted the ship-hold **scoped to
  this arm**, left D5 **open** ("not yet — wait for the arm"), and told me to leave the unrelated
  uncommitted `trinity/` documentation edits alone. All three honoured: `§7.1` is untouched, the arm
  is a **patch** (`hpc/b14/k10_arm.patch`) rather than a change to `main`, and I committed nothing
  from `trinity/`.
  **Implementation, and it got smaller than Batch 16 predicted.** `get_phii_k10` +
  `_k10_front_radius`, with `get_phii_c3c` aliased to the new helper so **not one call site changes**
  and reverting the arm is deleting a single line. `scipy.optimize` was already imported, so no new
  dependency. As registered, G16.3's "signature change" consequence is **retracted**: every input to
  `get_effective_bubble_pressure` is in `params` and that function lives in the same module, so the
  helper computes the ramped `P_conf` itself.
  ⛔ **G18.0 (BLOCKING) FAILED as written — 6.761e-02 against a 1e-10 bar — and is recorded failed.**
  The diagnosis is the finding: the error is **entirely the `P_conf` source**, proven by the
  per-phase split — `P_conf` rel err is **exactly 0.000e+00 in implicit and momentum** and
  6.817e-02 / 5.877e-03 in energy / transition. Batches 16 and 17 *recovered* `P_conf` from stored
  columns; production *recomputes* it from `Eb` and a freshly solved `R1`. They agree exactly where
  recomputation is trivial and diverge inside the `dt_switchon` ramp window and through the
  `Eb → Pb` reconstruction. **Production is right and my screens were the approximation** — Batch 17
  had already flagged this qualitatively, and G18.0 turned it into a number. This is precisely what a
  per-call gate is for: without it the arm would have been measured against a comparator nobody had
  checked.
  ✅ **G18.0′ (amended, on the G8.4′ precedent) PASSES at 1.005e-12** — same bar, same quantity, with
  the one input the screen could not reproduce held fixed. So the **production closure is the object
  Batches 16/17 validated**; the amendment isolates the closure rather than weakening it.
  ✅ **The arm runs.** `SC` to `stop_t` 0.01: 114 snapshots, 338 s, **zero distress lines**. And
  G18.1's contract change is live and correct — `P_HII > 0` on **97/97 energy** and **17/17
  implicit** rows where C3c returns exactly 0.0, with median `P_HII/Pb` = 0.8392 in energy (the
  ramped `P_conf` below the un-ramped `Pb`: the D-ramp being honoured) and 1.0008 in implicit.
  `test/test_phii_c3c.py` fails under the arm **by design**; recorded broken, not re-baselined.
  ⚠️ **Two things worth carrying forward.** (1) G18.0's diagnostic **bounds Batch 17's offline
  error**: momentum/implicit exact, transition ≤0.59%, **energy ≤6.8%** — so §Batch 17's G17.4 energy
  column reads ±7% and is corrected in place. (2) The local interpreter is **Python 3.8.8**, below
  trinity's stated ≥3.9, so `run_batch.py`'s post-run reporting raises on `Path.is_relative_to` and
  the walltimes CSV is skipped; run outputs are written before the raise and are unaffected. Helix
  activates the `trinity` env so it should not bite there, but it is a real wart on this machine.
  **Next is the ladder and nothing else** — G18.2/18.3/18.4/18.5 all need runs.
  `hpc/b14/sync.sh` now defaults to `ARMS="baseline k10"`; G18.5 closes Batch 17's ED-phase dust
  coverage gap for free from the arm's own output. D5 stays open until G18.3 reports.

- **2026-08-29 (maintainer asked me to check Batch 13's "cancellation" claim — it is WRONG, and
  retracted)** — I had flagged the wording as loose while explaining K10 vs C3c; the maintainer
  asked for it to be checked rather than left as a hunch. Correct instinct: it is checkable by
  algebra on committed data and should never have stood on plausibility. `P_HII` is linear in the
  density and C3a's density is `sqrt(Qi_abs/(χ_e α_B V))`, so each correction is a clean
  multiplicative factor — `f_volume = sqrt(V_cav/V_layer)` and `f_dust = sqrt(recomb/Qi_abs)` —
  and "opposite sign" is just whether one exceeds 1 while the other does not.
  **Verdict: both clauses fail, in different phases.** In **momentum, the regime the claim
  explicitly named, the corrections push the same way on 17/17 rows** (0.3862 × 0.6269 = **0.2506**)
  and therefore compound; transition compounds on 14/21. In **energy and implicit the signs ARE
  opposite** — so the claim is directionally right there — but `f_volume` is **31.98** and **11.49**
  against `f_dust` 0.497 and 0.907, netting ×15.4 and ×10.3, which is a 10–15× inflation rather than
  anything "of similar size".
  **The mechanism the claim missed** is one this workstream had already found: the volume correction
  changes sign at `R_IF/R2 = 2^{1/3} ≈ 1.26`, where the cavity-excluded layer volume equals the
  cavity volume. Measured `R_IF/R2` is 1.000 / 1.003 / 1.502 / 1.975 across the four phases, so the
  thin-layer phases sit below break-even and invert — the same thickness-dependent sign Batch 10
  recorded for the geometry correction. A claim about signs was made without checking the one
  quantity that sets them.
  **And it does not explain what it was invoked to explain.** C3a corrected for *both* errors gives
  momentum `P_HII/Pb` = **1.545**, while K10 gives **6.333** — **4.10× apart**. So K10 landing near
  the shipped drive is not a cancellation story; **K10 is a different closure** (pressure-equilibrium
  density, plus the `(R_i/R2)²` area amplification), not C3a with two fixes. The empirical
  observation survives — C3c and dust-corrected K10 agree to 10–15% here — but it is now recorded as
  **unexplained**, which is the honest state.
  🔑 **One genuine identity fell out**, and it is worth keeping: corrected-C3a and the **K5b profile
  form** both give 1.545 because they are algebraically the same quantity —
  `n_rms/n_cav = sqrt[(recomb/Qi_abs)(V_cav/V_layer)] = f_dust · f_volume`. **K5b IS C3a corrected
  for volume and dust.** That unifies G9.4's profile form with this decomposition and explains a
  number that has been quoted since Batch 9 without a mechanism. ⚠️ Stated as an identity, NOT as
  independent corroboration — the two routes agreeing to the printed digit is what the algebra
  requires, not evidence.
  ⚠️ Also noted: `data/b9_layer_density.csv` predates `layer_density_check.py`'s `pdrive_*` columns,
  so pressures here are joined from `b11_mass_ledger.csv` on `row_idx` (the Batch 14 join, with a
  per-row `|Δt| < 1e-4` guard). No `trinity/` source touched.

- **2026-08-29 (Batch 20 — K10 safety audit complete: UNSAFE as implemented, 1 CRITICAL + 4 MAJOR,
  and two of my own claims refuted)** — Maintainer, after the cancellation retraction: *"can you
  check other claims too so i know K10 is safe or unsafe?"* Four adversarial read-only slices, all
  registered in this doc **before any reported**. Every finding I could re-derive myself, I did.
  **CRITICAL — K10 has no photoionisation-only limit.** ✔ Verified by applying the arm patch in a
  clean worktree and running the suite: `test_phii_c3c_spitzer.py` **6 passed → 5 failed**. With the
  wind off `P_conf = 0`, the guard fires, and K10 returns **exactly 0.0 at every radius** where C3a
  gives the classical D-type pressure. Structural, not a bug: `drive ∝ P_conf^{−1/3}`, so the limit
  is **singular** and the guard converts a divergence into zero. ✔ The divergence is **already in the
  committed data** — momentum `drive/P_conf` 6.213 (B3M) vs 15.265 (B3MW01, `Lw`×0.1), ratio 2.457
  against the predicted `10^{1/3}` = 2.154. Batch 8 called that limit the one exact external anchor
  this family has. **This was my gate-design failure**: Batch 14's G14.2 protected exactly this for
  K5 and I did not carry a limits gate into Batch 18.
  **MAJOR — seam C present and worse than C3c's** (implied layer mass 2.4892× the shell at `t` = 1.5;
  1.628 vs C3c's 1.5638 against all available gas), **and the drive IS the geometry**, so the mass
  book cannot be fixed without cutting the drive 1.78×. **MAJOR — the per-segment freeze ratchet is
  re-armed** (✔ verified in source: frozen `snapshot.P_HII` vs live `press_bubble` in a `max`;
  ~8% median, 17% max staircase against the 0.55% term K10 exists to deliver) — a class §3 rates
  *"catastrophic at compact scale"*, with the compact config PRB untested. **MAJOR — coverage**: two
  trajectories of one cloud, to 1.5 Myr, against C3c's 13 configs with a fate table. **MAJOR (review
  hygiene) — `test_mu_audit_drift.py` now passes VACUOUSLY**: the patch is additive, so the assertion
  matches the dead `def` body while the live path matches nothing. A green test giving false
  assurance.
  ✏️ **Two of my own claims were refuted by the audit, and both are recorded as such.** (1) The
  `n_IF_Str` gate I registered as "a REAL suspected defect" is **MINOR** — `n_IF_Str == 0` on **0 of
  3490** rows across nine committed runs. (2) My `k·(hi−R2) < 1e-8` guard **tests the wrong
  quantity** — cancellation is governed by `k·R2`, and the guard **never fires for any nonzero dust**
  (0 times, 1× down to 1e-20×). Slice 3's proposed remedy (raise the threshold) would not have worked
  either; the fix is an `expm1` refactor. Also corrected: the G18.0 `P_conf` discrepancy is confined
  to **2 of 156 energy rows**, narrower than §Batch 18's "energy ≤6.8%" caveat.
  ✅ **What survived:** seam A genuinely absent; the composition mapping settled (2.22e-16) with
  non-negativity now **proven** rather than sampled; the dust closure validated where checked; G13.3's
  `χ_e` diagnosis confirmed **and strengthened** to an exact per-row form (1.04e-14 on 59 rows); units
  clean; cost invisible (0.6% of one shell slice); and **no K10-specific jump at the
  transition→momentum handover** — a registered worry that did not materialise.
  **Disposition: Batch 18 stays ⛔ HELD.** The blocking items are physics, not typos: whether a
  closure with **no photo-only limit** is acceptable for trinity, and whether the freeze ratchet
  forces calling the helper live inside the ODE. Two one-line code fixes (`expm1`; `try/except` around
  `brentq`) are prerequisites but not sufficient. **Cheapest real progress is not the arm** — re-point
  the existing offline screeners at the other core-6 trajectories: no new physics, no `trinity/`
  change, and it closes the density/mass/sfe axes. No `trinity/` source touched this visit.

- **2026-08-29 (maintainer challenges the CRITICAL finding — half right, and the half that is right
  makes the finding stronger)** — Maintainer, on being told K10 breaks the photo-only limit: *"what
  if a photoionised limit cannot be reached in our Weaver-like wind-driven bubble?"* Correct on the
  literal point: `P_conf` is strictly positive in all four phases, so `P_conf = 0` is unreachable and
  `test_phii_c3c_spitzer.py`'s failures are **not a runtime hazard**. As a runtime severity, that
  finding is **downgraded**. But the challenge asks the better question — how far into the
  `P_conf^{−1/3}` divergence do the runs already sit — and the answer, measured on committed data
  (`harness/k10_domain_check.py`, `data/b20_domain.csv`), is: **all the way in.** K10's ionisation
  front lies beyond the shell the ODE produced on **18/18** B3M momentum and **43/44** energy rows,
  and beyond `rCloud` (4.999 pc) on **100% of driving rows in both configs** — median `R_i` = 5.55×
  the cloud radius in B3M momentum, **max 72.7 pc in a 5.0 pc cloud (14.5×)**. The wind ladder is
  moving toward the singularity, not away: `drive/P_conf` 6.333 → 15.265 across one decade (ratio
  2.410 vs the predicted 2.154), extrapolating to ≈29.4 and `R_i` ≈ 42 pc at the **registered but
  never-run** `B3MW001`. **Re-labelled MAJOR-domain rather than CRITICAL-runtime**, and it is now
  visibly the *same* defect slice 2 found as mass over-subscription and slice 1 found as
  `R_i > rShell` — three independent slices converging on one thing: the CEM's geometric
  amplification is unbounded and trinity is already outside its domain. **The old framing was
  dismissible and the maintainer was right to push on it; the measured one is not.**
  ⚠️ **I also downgrade my own "re-point the screeners at the core-6" recommendation.** It was the
  right advice when the picture was "sound scheme, thin coverage". It is the wrong advice now: four
  more configs would measure the same hole in more places, cost hours of wall-clock, and change no
  decision on the table. The maintainer's question caught that before the compute was spent. The
  live decision is instead **whether K10's front should be capped** — at the shell edge, at `rCloud`,
  or by available mass — which slice 2 has already priced at a **1.78× cut to the drive**
  (`drive/P_conf` 9.770 → 5.49). No `trinity/` source touched.

- **2026-08-29 (maintainer picks O1; three of Batch 20's five blockers close, two do not)** —
  Maintainer, after Batch 20 priced the four candidate remedies: *"let's do O1, use the shell solve's
  own front."* Gates registered and committed before implementation, per §0.
  **The change is a deletion, not an addition.** `rho = (shell_props.R_IF / R2)**2`, and
  `_k10_front_radius` **goes away entirely**. `R_IF` is a first-class field on `ShellProperties`
  (`shell_structure.py:39`, set at `:227`), so this is a read. Removing the front solve closes slice
  1's **F1** (guard tested `k·(hi−R2)` when cancellation is governed by `k·R2`, so it never fired),
  **F2** (float64 bracket failure → unhandled `ValueError`, or a silently 8.9%-wrong root at low
  metallicity) and **F4** (`ZeroDivisionError`/`OverflowError`) *by construction* — there is no
  solver left to be fragile. It also makes `shell_props` load-bearing again, retiring slice 1's F7
  and giving the vestigial `n_IF_Str > 0` gate something real to gate (**B11.E**).
  **Measured (189 rows).** **G21.1 PASS**: front inside the shell on **140/140** rows, where the
  Batch 18 form was outside on 18/18 momentum and 43/44 energy. **G21.2 PASS**: implied layer mass
  ≤ shell mass on **every row**, worst 0.9407, against the old max of **2.4892** — **seam C is
  closed**. **G21.4**: the confined excess **survives at +0.475%** (about half the old +0.67%), so
  my pre-registered worry that Batch 16's first-order-term argument would evaporate did **not**
  materialise; it stands in reduced form.
  ⚠️ **G21.3's disclosed prior missed by 16%** — I registered 3.274… no: I registered **3.901** from
  Batch 20's pricing and measured **3.274**. Recorded as a miss. Not an error: Batch 20 took the
  median over `b9`'s own 17 momentum rows while the screen joins `b17`'s 21 rows to the nearest
  front, and the two span different `t`.
  **The magnitude story is the headline for an arm.** The confined branch is left essentially
  untouched (O1/shipped **1.001** energy, **1.005** implicit — unlike the Batch 18 form, O1 does not
  disturb the branch §6b found exactly self-consistent), while the driving branch is **roughly
  halved against shipped C3c**: ×0.494 in B3M momentum, ×0.325 in B3MW01. That is a large and
  measurable change.
  ⛔ **What O1 does NOT fix, verified under the patch rather than asserted (G21.5):**
  `test_phii_c3c_spitzer.py` is **still 5 failed / 1 passed** — the drive is still ∝ `P_conf`, so
  there is still **no photoionisation-only limit**; O1 fixes the geometry, not the singularity. The
  **freeze ratchet** is untouched. Coverage is still two configs of one cloud. And D5's magnitude
  question is **not** resolved — 3.274×`P_ram` is still photoionisation-dominated, by ~3× instead of
  ~7×.
  🔑 **Batch 17 is now moot as machinery**: the shell solve's `R_IF` already carries dust, so the
  closure's own dust model is not shipped under O1. Batch 17's value is retroactively *validation* —
  it showed the closure's dust matched the shell solve's to 5.6%, which is now an argument for
  reading `R_IF` rather than a component of the code. `hpc/b14/sync.sh` now defaults to
  `ARMS="baseline k10_o1"`. Batch 18's `k10_arm.patch` is **superseded** but kept for provenance.
  No `trinity/` source touched on `main`.

- **2026-08-29 (maintainer's diagnosis of C3c, and a revisit of Batch 14's verdict on K5 that I owe)**
  — Maintainer: *"the current version produces that but it uses cavity volume without the real cavity
  density. But is there a way to meet in the middle?"* The diagnosis is sharper than the one in this
  doc: C3c's defect is not "wrong volume" or "wrong density" separately — it is a **fictitious
  density evaluated over a real region that physically cannot host it.** Batch 20's cavity check
  measured the actual cavity gas as **4–5 orders of magnitude too thin to absorb the photons C3a
  spends there**, on the `n²` term alone.
  **The middle the maintainer is reaching for exists, and it is K5b.** Replacing the fictitious
  density with the code's own profile density over the code's own layer volume gives exactly C3c's
  *structure* (a thermal pressure that composes additively) with none of its fiction. Batch 19 already
  proved this is not a new candidate: `n_rms/n_cav = f_dust · f_volume` **identically**, so **K5b IS
  C3c corrected for both volume and dust**. B3M momentum: **1.545×`P_ram`**.
  ⚠️ **But K5b and O1 are NOT two points on one axis, and averaging them would be picking a number
  rather than a model.** Measured on the same B3M momentum rows, both expressed against `P_conf`:
  K5b amplifies by the **density** ratio (`n_rms/n₀` ≈ 1.64), O1 by the **area** ratio
  (`(R_IF/R2)²` = 3.901). They differ by **2.38×**, and that gap **is D5** — "what does the ionised
  layer transmit?" — not a numerical discrepancy to be split.
  ✏️ **Revisit I owe on Batch 14.** Batch 14 rejected K5 on two grounds: (a) it is `Pb`-slaved
  (∝`Pb^{1.0}`, gain ≈1.5), and (b) the additive composition then gives ≈2.5×`P_ram`, structurally
  reminiscent of the pre-C3c double count. **Ground (a) no longer stands as written**: the same-day
  D5 reframe established that pressure-slaving is *physics* when it is structural — the ionised
  layer's density genuinely is set by the `nShell0 ∝ Pb` boundary condition, which Rahner+2017 models
  as quasi-hydrostatic and Lancaster+2025 assumes explicitly. K5b's `Pb`-dependence is that same
  statement, not a relabelling. **Ground (b) survives, but it is D5's additivity question, not a
  verdict against K5.** Net: **K5b should be treated as live again**, not closed, and Batch 14's §7.1
  row is corrected accordingly.
  🔑 **A con of O1 that belongs on the record.** O1 borrows *half* of Lancaster's geometry: it applies
  the `(R_i/R2)²` force amplification, which in Lancaster exists because the neutral shell sits **at**
  `R_i` (his eq. 28 drives `R_i` with `M_sh = (4π/3)ρ̄R_i³`), while trinity tracks its shell at `R2`
  with the ionised layer **inside** `shell_mass`. Slice 3 sized that mismatch at a factor 13.7 in the
  CEM's own inertia bookkeeping. So O1 is geometrically self-consistent about *where the gas is* and
  not about *where the inertia is* — the second half is K8/K9, which the maintainer deferred.
  **The honest range, B3M momentum, all ×`P_ram`:** transmit-only **1.0** (K3/C1) · K5b **1.55** ·
  O1 **3.27** · K10 **6.33** · shipped C3c **7.10**. No `trinity/` source touched.

- **2026-08-29 (exploratory: can the bubble's real density inform `P_HII`? — the maintainer's own
  prediction confirmed, a `bubble_mass` defect localised, and a temperature I got wrong)** —
  Maintainer: *"can we try out the bubble profile method (since we have bubble mass and volume we can
  easily get density)… keep in mind that in transition and momentum phase there will be no bubble
  since it slowly gets compressed into R2 because Pb → 0."* Measured before designing anything
  (`harness/bubble_density_probe.py`, `data/b22_bubble_density.csv`), using the **shocked-wind
  volume** `(4/3)π(R2³ − R1³)` rather than the full sphere.
  ✅ **The prediction is exactly right, and quantified.** `R1/R2` runs 0.286–0.874 (energy),
  0.150–0.273 (implicit), then climbs to **0.9991** in transition — the bubble volume falls to
  **0.26% of the sphere** — and in momentum `R1/R2 = 1.0000` with `Eb = 0.0` exactly, so
  **`V_bub = 0` and the density is UNDEFINED on 17/17 momentum rows**. Any scheme built on bubble
  density has a hard singularity precisely where the maintainer said it would.
  ⛔ **And there is a worse problem, which localises a defect B11.0 only named.** `bubble_mass`
  evolves properly in energy (44 distinct values) and implicit (34 distinct), then **FREEZES at
  99.6429 for the whole of transition and momentum** (1 distinct value across 38 rows). So it is not
  merely that the volume vanishes in momentum — from the **transition entry onward the mass is stale
  too**, and a frozen mass inside a collapsing volume gives a density that is wrong in both factors.
  This is a **trinity defect independent of any `P_HII` scheme** and belongs to whoever owns the
  bubble solver.
  ✏️ **CORRECTION I OWE: the maintainer's `T ~ 1e7` was right and my earlier number was wrong.** In
  Batch 20's cavity check I reported the implied bubble temperature as **1.8e5–1.3e6 K** and said it
  sat below the ~1e7 the maintainer assumed. That used the **full sphere** `(4/3)πR2³`, which is the
  correct volume for the question that check was asking (C3a's own cavity premise) but the **wrong**
  one for the physical bubble, which occupies only `R1 < r < R2`. With the correct volume:
  **T_implied = 2.157e7 K (energy), 8.0e6 K (implicit)** — the maintainer's intuition, to the order.
  (Transition falls to 5.2e4 K, but that is the frozen-mass artefact above, not physics.)
  ✅ **What the probe DOES deliver, and it is worth keeping.** With the right volume, the cavity's
  own recombination consumes only **1.19e-5 (energy) / 1.66e-4 (implicit) / 6.18e-4 (transition)** of
  `Qi`. That is an *independent, correctly-volumed* confirmation that the wind cavity is transparent
  to ionising photons — so `shell_structure.py:120`'s `phi0 = 1` is right, and Batch 20 slice 2's
  "seam A is absent" verdict is strengthened from a second direction.
  **Verdict on the method: it cannot carry `P_HII`.** It is well-defined and physically sensible only
  in energy/implicit, degrades through transition on a stale mass, and is undefined in momentum. Using
  it would require a phase branch — reintroducing exactly the discontinuity O1 was chosen to remove.
  Recorded as **exploratory and closed**, with its two by-products kept: the transparency
  confirmation, and the `bubble_mass` freeze localised to the transition entry. No `trinity/` source
  touched.

- **2026-08-29 (the `bubble_mass` freeze: mechanism found, consumer found, impact sized — and it is
  NOT a dynamics bug)** — Maintainer: *"let's fix the bubble_mass freeze first. but im not sure what
  to set it to."* Diagnosed before proposing a value.
  **Mechanism.** `bubble_mass` is produced only by `bubble_luminosity.get_bubbleproperties_pure`,
  which is called from `run_energy_phase.py:181` and (via `get_betadelta.py:471,573`) from the
  implicit phase. **`run_transition_phase.py` and `run_momentum_phase.py` import only
  `get_bubbleParams`, never `bubble_luminosity`** — so the bubble structure is never solved there and
  `bubble_mass` is a stale carry-over from the last implicit step. (The `ADAPTIVE_MONITOR_KEYS` lists
  that also mention it are adaptive-stepping monitors and are unrelated.)
  **Consumer.** Exactly one, `shell_structure.py:268`:
  `grav_ion_m_cum = np.cumsum(grav_ion_m) + mBubble` — the mass enclosed inside `R2`, used for the
  shell's own gravity profile. So the semantically correct quantity is **"mass enclosed within
  `R2`"**, not "mass of the shocked bubble".
  **Impact, measured (B3M):**

| phase | `bubble_mass`/`shell_mass` | frozen value | physical free-wind mass `2·L_mech·R2/v³` | ratio |
|---|---|---|---|---|
| energy | **35.89%** | 0.0106 | 0.0003 | 0.024 |
| implicit | 0.49% | 13.31 | 0.0117 | 0.0009 |
| transition | **0.0996%** | 99.6429 | 0.0496 | 0.0005 |
| momentum | **0.0995%** | 99.6429 | 0.1156 | 0.0012 |

  **Two things follow.** (1) In momentum `R1/R2 = 1.0000` and `Eb = 0.0` exactly, so there is
  **provably no shocked-wind region**; the only mass inside `R2` is free wind in transit,
  **0.116 Msun**, and the frozen 99.64 is **860× too large**. (2) But `bubble_mass` is only **0.1% of
  `shell_mass`** in both affected phases, so the error in the gravity term is ~0.1%. **The freeze is a
  correctness and hygiene defect, not a dynamics bug** — and it is *large* only in the energy phase
  (35.9% of the shell), which is precisely where it is computed correctly.
  ⚠️ **It did corrupt two of this workstream's own diagnostics** — Batch 20's cavity check and the
  2026-08-29 bubble-density probe both read `bubble_mass`, so their transition/momentum numbers
  inherit the stale value. Both already carry that caveat.
  **Candidate fixes, for the maintainer:** (a) **momentum only** — set the enclosed mass to the
  free-wind mass `2·L_mech·R2/v³` (all inputs in `params`, one line, provably correct there since
  `R1 = R2`); (b) **transition too** — same formula, principled in the `R1 → R2` limit but
  under-counting at transition entry where `R1/R2 = 0.18` and a real bubble still exists; (c) **run
  the bubble solve in transition** — the "right" fix, but expensive and of uncertain convergence as
  `R1 → R2`; (d) **rename the field to what its consumer wants** (`mass_enclosed_R2`) — clearest, but
  touches the registry, snapshots and readers. ⚠️ Any of (a)-(c) changes the shell gravity by ~0.1%
  and is therefore a **dynamics change**: under CLAUDE.md rule 5 it needs its own gate and a full-run
  equivalence, and it is **outside** the K10-arm-only ship-hold lift. Not implemented; awaiting the
  maintainer's choice of value and scope. No `trinity/` source touched.

- **2026-08-29 (bubble_mass fix — option (a), momentum only: gates registered BEFORE implementing)** —
  Maintainer chose option (a). ⚠️ **Scope note:** this is a **production `trinity/` change and is NOT
  part of the K10 arm**, so it sits outside the 2026-08-28 arm-only ship-hold lift; it proceeds on the
  maintainer's explicit 2026-08-29 instruction. ⚠️ **Housekeeping:** this is a *bubble-solver* defect,
  not a `P_HII` one — it is recorded here because it was found here, and §0's one-doc rule is about
  not sprawling the `P_HII` effort. If it grows beyond this fix it should get its own workstream.
  **The change.** In the momentum phase `R1 == R2` and `Eb == 0` exactly, so no shocked-wind region
  exists and the only mass inside `R2` is free wind in transit. Set `bubble_mass` to
  `M_fw = 2·L_mech·R2 / v_mech³`, which is `Mdot·R2/v` under **`pRam`'s own convention**
  (`Mdot = 2 L/v²`, `get_bubbleParams.py:286`) — so the fix inherits the code's existing wind
  convention rather than introducing a second one. Both momentum shell-solve sites are covered
  (`run_momentum_phase.py:628` and `:894`); `params['Lmech_total']`/`['v_mech_total']` are current at
  both, refreshed by `updateDict(params, feedback)` at `:578` and `:890` (verified: `SPSFeedback`
  carries both fields).
  **Gates.**
  - **GB.0 — momentum only (BLOCKING).** `bubble_mass` in energy / implicit / transition must be
    **bit-identical** to the current code. *Falsifier:* any change outside momentum.
  - **GB.1 — the value uses the code's own convention.** The helper lives beside `pRam` and its
    `Mdot` is `pRam`'s. *Falsifier:* any second wind convention introduced.
  - **GB.2 — magnitude (measurement).** Momentum `bubble_mass` 99.6429 → ~0.116 Msun on B3M, i.e.
    the enclosed-mass term drops from 0.0995% of `shell_mass` to ~0.0001%.
  - **GB.3 — full-run equivalence (CLAUDE.md rule 5, BLOCKING before adoption).** Matched-`t` ΔR2 on
    B3M (the only core config reaching momentum) and B3MW01, separate processes. **Pre-registered
    expectation: ≤0.1%**, since the corrected term is ~0.1% of the enclosed mass. *Falsifier:* >1% ⇒
    the term matters more than the sizing implies and the fix needs re-thinking, not just re-running.
  - **GB.4 — goldens.** Any test movement reported with the before/after value, per D4's discipline.
    Momentum-reaching goldens may shift.

- **2026-08-29 (bubble_mass fix RESULT — all gates pass, and the equivalence run explains why)** —
  Two B3M arms to `stop_t` = 1.0, separate processes in parallel (C-3), pre-fix worktree `cc53a656`
  vs the fix. Both reached all four phases (87 energy / 68 implicit / 42 transition / 23 momentum),
  so nothing is VOID. Ledger: `data/gb3_bubblemass_ledger.csv`.
  - **GB.0 ✅ PASS — momentum only.** From the runs' own output: implicit `bubble_mass` 1.2952…99.6429
    in **both** arms, transition 99.6429 in **both** (still stale, as designed). Only momentum moves.
  - **GB.1 ✅ PASS.** `mass_freeWind` = `Mdot·r/v` exactly, with `Mdot = 2L/v²` — `pRam`'s own
    convention, verified numerically against `ρv²` to machine precision. No second convention.
  - **GB.2 ✅ measured.** Momentum `bubble_mass` **99.642929 → 0.0580–0.1603 Msun**, bracketing the
    0.116 predicted from committed data.
  - **GB.3 ✅ PASS, and stronger than the bar.** Pre-registered expectation was ΔR2 ≤ 0.1%. Measured:
    **bit-identical.** `compare_trajectories.py` gives `dR2_max` = **0.000%**, `dR2_end` = 0.000%,
    fate `stopping_time → stopping_time`; direct row comparison finds **0 of 220 rows** with any `R2`
    difference at all, `R2_end` = 14.0584340349 in both.
  - **GB.4 ✅ PASS trivially** — bit-identical trajectories cannot move a golden, and the full suite
    already showed 1098 passed with the 5 failures verified pre-existing against clean code.
  🔑 **Why it is bit-identical, traced rather than assumed.** `bubble_mass` reaches the dynamics
  through nothing: `shell_structure.py:268`'s `grav_ion_m_cum` feeds only `shell_grav_r` /
  `shell_grav_phi` / `shell_grav_force_m`, and the only consumers of those outside
  `shell_structure.py` are `registry.py` (schema) and `dictionary.py` (snapshot serialisation). The
  equation of motion's gravity is the separate `F_grav = G·mShell/R2²·(mCluster + 0.5·mShell)`
  (`energy_phase_ODEs.py:218`). **So `shell_grav_*` is diagnostic-only and the fix is pure hygiene
  with zero dynamics risk.**
  ✏️ **Which means my own risk framing was over-cautious.** I called this "a dynamics change needing
  a full-run equivalence under rule 5". It is not a dynamics change. Running the gate was still the
  right call — the point of rule 5 is that this is now **proven** rather than assumed, and the
  proof cost one 8-minute parallel pair. But the caution should be recorded as having been
  unnecessary, not quietly dropped.
  **What the fix actually buys:** the *reported* `bubble_mass` in momentum is now the true enclosed
  mass instead of a value 860× too large. That matters for anything reading the output — including
  two of this workstream's own diagnostics, which it corrupted.
  ⚠️ **Transition remains stale by design** (99.6429 throughout) and is documented, not fixed:
  the bubble is real at transition entry (`R1/R2` = 0.18) and gone by exit, so the free-wind formula
  would under-count there. Option (b) remains open if the maintainer wants it.

- **2026-08-29 (housekeeping scan — four adversarial audits over this doc, the code, the harnesses and
  the siblings; what they found in MY OWN record)** — Maintainer asked for a scan before closing out.
  Four read-only audits. **The harnesses came back clean** — no broken script, every plan number I
  sampled reproduces exactly from its committed CSV — and the **production `bubble_mass` fix audited
  accurate in every citation**. The debt was in the record, and most of it was mine.
  ⛔ **Two corrections I claimed to have made and had not.** (1) The 2026-08-29 entry said "Batch 14's
  §7.1 row is corrected accordingly" — **it was not**; the K5 row still asserted the withdrawn
  `Pb`-slaving rejection, in a table this doc calls "the single list of live options". Now corrected.
  (2) An entry said Batch 17's G17.4 energy column "is corrected in place" — **it was not**; the ±7%
  annotation was never added, and the figure has since been narrowed to **2 of 156 rows**. Both fixed.
  This is the exact failure mode §0 exists to prevent, and claiming a fix is worse than not making it.
  **Corrected in place:** §7.1's K5 row (ground (a) withdrawn, K5b live again) and K8 row (tier
  **A → S**, deferred not pending); the C-0 tier legend and closing line (rev2 landed 2026-08-18);
  Batch 18's status and ledger row (**HELD *and* SUPERSEDED by Batch 21**, ladder owed against the O1
  patch); Batch 20's CRITICAL → **MAJOR-domain** relabel propagated to the heading, verdict and
  bullet, and its finding count reconciled to **4 MAJOR + 1 MODERATE**; Batch 13's withdrawn "+34% not
  23.4%" sentence marked; the "energy ≤6.8%" caveat narrowed wherever it appeared; "2 configs of 20"
  → **13**; and **G21.2's coverage overstatement** — it said "every phase and both configs" when
  `b21_o1_screen.csv` has **no B3MW01 ED rows at all**, so those claims are B3M-only.
  **§8.1 gained rows for Batches 13 and 20 and for the `bubble_mass` fix** — §8.1 is titled "the one
  source of truth" and contained no trace of the audit that made K10 unsafe. **§8.3 gained**
  `b20_domain.csv`, `b22_bubble_density.csv`, `gb3_bubblemass_ledger.csv` and — closing a gap open
  since the carve-out was written — **`LITERATURE_ASSESSMENT.md` itself**, whose C-0 condition 4
  ("indexed in §8.3") had never actually been met.
  🔑 **One systemic finding worth more than the individual fixes.** Every harness here defines
  `med(v) = sorted(v)[len(v)//2]` — an **upper order statistic, not a statistical median**. On
  even-length row sets it differs from `numpy.median` by ~1%, and that is exactly why **7.095 and
  7.1646** both circulate in this doc for C3c's momentum drive, and why B3M's K10 momentum
  `drive/P_conf` appears as **6.213 / 6.333 / 5.415** from three different row-set joins. None of the
  numbers is wrong; the convention is just undeclared. Anyone re-deriving a quoted figure with a true
  median will not reproduce it. **Declared here rather than silently re-fitted.**
  **Code/script fixes:** `k10_percall_equivalence.py` guarded against the O1 arm (its alias check
  passed but it then called the deleted `_k10_front_radius`); `k10_domain_check.py`'s `7.095`
  re-attributed to its real source; the Batch 18 arm patch given a **SUPERSEDED-AND-HELD banner**
  naming its four known-wrong statements, and `run_arms.sh` now **refuses** `k10` without an explicit
  force; the `b14_` labels that would have misfiled Batch 21 output as Batch 14 renamed to `phii_`;
  the O1 patch's misquote of G16.3 (`0.33-1.07x` → **0.330-0.995x**, since a ratio > 1 inside the ramp
  window is impossible by construction) fixed, and its "known not fixed" list extended with the
  vacuous `test_mu_audit_drift.py` pass. `mass_freeWind` now states its steady-state assumption, and
  `run_transition_phase.py` carries an in-source marker that its `bubble_mass` is stale by design.
  ⚠️ **`data/gb3fix_walltimes.csv` was committed EMPTY** — a stamp and a comment, no header, no rows.
  Moved to `docs/dev/to-be-removed/`. Its baseline counterpart never existed (that arm ran in a
  worktree whose `data/` went with it).
  **Left open, deliberately:** `data/b9_production_branch_screen.csv` has **no C-6 stamp at all**
  (1.5 MB, from the unreferenced `harness/screen_production_grid.py`, which has no argparse and
  hardcodes its output path) — a pre-existing violation predating this session, and a decision for the
  maintainer: stamp and regenerate, or retire the cluster. Also left: `data-new/README.md` names a
  `harness/make_manifest.py` that does not exist and still carries a placeholder "Cut commit";
  `g34_golden_rebaseline.csv`'s stamp is prose rather than machine-parseable; three harness docstrings
  have cosmetic inaccuracies; and this doc's H1 still says branch `bugfix/phii-pt1`.

- **2026-08-29 (DECISION, delegated by the maintainer: is a closure with no photoionisation-only
  limit acceptable? — YES for O1, conditionally; and the reason is a measurement, not a preference)**
  — Asked to decide rather than defer. The answer turns on a distinction the earlier framing missed,
  so I measured it first: **O1 does not inherit K10's divergence.**

| | growth in `drive/P_conf` per wind decade (B3M → B3MW01) |
|---|---|
| Batch 18 K10 (own analytic front) | **×2.784** momentum, ×3.056 transition |
| `P_conf^{−1/3}` divergence predicts | **×2.154** |
| **Batch 21 O1 (shell solve's front)** | **×1.461** momentum, ×1.700 transition |

  K10 grows **faster** than the divergence law — it is running into the singularity. **O1 grows
  markedly slower**, because its front is the shell solve's and is bounded by the shell's own mass
  (G21.1: inside the shell on 140/140 rows). So under O1 the drive is `P_conf × (bounded ρ)`, which
  as `P_conf → 0` goes **gracefully to zero rather than diverging**. The pathology that made Batch
  18's form dangerous is gone; what remains is a *different and much better-posed* defect.
  🔑 **Restating the defect correctly: O1 is not missing a limit, it is missing a FLOOR.** Physically
  the photoionised gas cannot push less than its own thermal pressure, so as the wind weakens the
  drive should approach the Spitzer value — a finite, non-zero floor — instead of following `P_conf`
  to zero. That floor is a quantity this workstream has already measured: it is **K5b**, the real
  profile's layer pressure, **1.545×`P_ram`** on B3M momentum. A composite
  `max(P_conf·ρ, P_layer)` would hold both limits, and is the principled version of the "meet in the
  middle" the maintainer asked about — not an average of two schemes but a floor under one.
  **DECISION: acceptable, conditionally.** Grounds: (1) `P_conf = 0` is **unreachable** in trinity —
  every phase carries a strictly positive confining pressure, so the failing fixture is artificial;
  (2) trinity is a **wind-driven bubble code by construction**, and both Rahner+2017 and
  Lancaster+2025 assume confinement explicitly, so a confinement-requiring closure is in keeping with
  the lineage rather than a departure from it; (3) Batch 8's Spitzer anchor existed to validate
  **C3a's free normalisation** — O1 has no free normalisation to anchor, inheriting the shell solve's
  boundary condition and front, so there is nothing for that anchor to check; (4) the measurement
  above shows the approach to the singularity is **suppressed, not merely unreached**.
  **The condition, and it is not optional: the domain must be declared AND guarded, because the
  failure is silent.** O1 progressively **under-drives** as the wind weakens, and **`B3MW001`
  (`Lw`×0.01) is in the registered matrix** — a run there would quietly receive too little
  photoionised drive with nothing in the output saying so. Cheapest sufficient guard: emit a
  diagnostic whenever `P_conf·ρ` falls below the ionised layer's own thermal pressure, i.e. whenever
  the model is below its own physical floor. That is one comparison against a quantity the shell
  solve already computes.
  ⚠️ **Scope of this decision, stated so it is not over-read.** It says the missing limit is
  acceptable *as an approximation inside trinity's declared domain*. It is **not** a claim that the
  physics is right in general, **not** an adoption of O1 (D5 stays open — the maintainer ruled "wait
  for the arm"), and **not** a release of the arm: the **freeze ratchet** and the **two-config
  coverage** are untouched by this and remain blocking. What would reverse it: evidence that a
  science case needs the weak-wind corner quantitatively, or a measured trajectory where the
  under-drive changes a fate.

- **2026-08-29 (Batch 21 ARM RUN — G18.2/G18.3 measured on Helix; the prediction holds on BOTH
  branches, and one config changes phase structure)** — Ladder run on Helix, sweep
  `phii_arm_20260829_112826Z`, both arms at the pinned `cce8c924`, 10 runs, `stop_t` 1.5.
  **C-7 provenance ✓** — `k10_o1_applied.diff` is byte-identical to the committed
  `k10_o1_arm.patch` apart from the `index` metadata line, which is stale in the committed copy
  because a `+` line was hand-edited during the housekeeping pass without regenerating it;
  `git apply` matches by context, so the code that ran **is** the committed content. The baseline's
  applied diff is 0 bytes, i.e. genuinely unpatched.
  **G18.2 ✅ PASS.** All 10 runs `ok`. Wall times `k10_o1`/baseline = **1.04–1.56×** against a 2× bar
  (B3M 563.4 vs 463.5 s; B3MW01 746.6 vs 695.7; F1HI 704.6 vs 453.1; F1LO 714.9 vs 688.4).
  **G18.3 — no fate changed on any config**, and the signed ΔR2 is the headline:

| config | phases reached (arm) | `R2_end` base → new | signed ΔR2 | verdict |
|---|---|---|---|---|
| B3M | energy>implicit>transition>momentum | 23.2527 → **20.8475** | **−10.34%** | OVER-BAR |
| B3MW01 | energy>implicit>transition>momentum | 7.7331 → **6.7112** | **−13.22%** | OVER-BAR |
| F1HI | …>transition>**momentum** (base stops at transition) | 0.5410 → 0.5561 | +2.79% | **PHASE-CHANGE** |
| F1LO | energy>implicit | 185.519 → 189.370 | +2.08% | WITHIN-BAR |
| SC | energy>implicit | 87.8889 → **92.5292** | **+5.28%** | OVER-BAR |

  🔑 **The two-sided sign is the result, and it is exactly what Batch 21 predicted, for the predicted
  reasons.** The configs that reach the **driving** branch shrink (B3M, B3MW01), because O1 cuts that
  drive (×0.494 and ×0.325 measured offline) — and the larger drive cut gives the larger radius cut,
  in the right order. The configs that never leave the **confined** branch grow (F1LO, SC), because
  O1 returns a small positive excess where C3c returns exactly 0.0. Both directions, both
  magnitudes, from one prediction.
  ⚠️ **New finding, and it corrects how G21.4 was read: the confined-branch excess is NOT dynamically
  negligible.** G21.4 measured it at **+0.475%** of `P_conf` and the batch treated that as a small
  term. Compounded over a full run it moves **SC's `R2` by +5.28%** — over the 5% bar, on a config
  that never reaches the driving branch at all. This is Batch 4a's lesson repeating ("early-window
  drive changes can retain double-digit ΔR2 downstream", F1LO 14.4%), and it means **O1 is not
  "essentially untouched" on the healthy branch** the way §Batch 21's `O1/shipped` 1.001/1.005 made
  it look. A ~0.1–0.5% drive change is a several-percent trajectory change.
  ⚠️ **F1HI changed phase structure: `energy>implicit>transition` → `…>transition>momentum`.** O1
  pushes it into a phase the shipped scheme never reaches (144 snapshots vs 135), while both arms
  still end `shell_collapsed`. This is the defect class §Batch 6 recorded slipping past on SDHS —
  *"a fate-only check does NOT catch this"* — and here the comparator **did** catch it, because
  `phases_base`/`phases_new` are now enumerated. Note it came with one of the **smallest** ΔR2 values
  (+2.79%): a magnitude bar alone would have missed the largest structural change in the ladder.
  ⚠️ **Two provenance gaps in the run artifacts, neither affecting the result.** The wall-time CSVs
  carry **`code unknown`** instead of a SHA (C-6: `code_version()` returned nothing inside the
  detached worktree), and **SC is absent from both** wall-time CSVs though it ran in both arms and
  appears in the ledger — the "rows merged across concurrent streams / last writer" merge dropped it.
  **Still owed for this batch:** G18.4 (continuity on O1's own trajectory — needs the seam ratios from
  the run dirs, which stayed on `/gpfs`) and G18.5 (ED-phase dust, which would close Batch 17's
  one-row coverage gap for free). **Still blocking regardless of this result:** the per-segment freeze
  ratchet and the two-config-of-13 coverage. D5 remains open.

- **2026-08-29 (Batch 21 arm — INDEPENDENT verification from the raw runs, on the maintainer's
  instruction to assume nothing; it caught two of my own errors)** — Maintainer: *"double check your
  inference with no assumption … independent without contamination."* So `harness/b21_arm_verify.py`
  reads only each run's own `dictionary.jsonl`, uses its **own** matched-`t` interpolation, and does
  **not** read the reduced ledger or import `compare_trajectories.py`. The ledger is used only as an
  after-the-fact cross-check.
  ✅ **V1 phase sequence — CONFIRMED.** F1HI really does gain a momentum phase under the arm
  (`energy>implicit>transition` → `…>momentum`); the other four sequences are identical.
  ✅ **V2 matched-`t` ΔR2 — reproduces the ledger** on an independent interpolation: SC 8.719/+5.280,
  F1LO 2.076/+2.076, F1HI 2.788/+2.788, B3M 10.344/**−10.344**, B3MW01 13.215/**−13.215** (ledger
  gave 8.718 / 2.076 / 2.790 / 10.344 / 13.215). The signs stand.
  ⛔ **V4 — MY EARLIER DISTRESS FINDING WAS A FALSE POSITIVE, and it was my own bug.** I reported
  "F1HI arm has MORE distress (68 vs 62)". The regex included `inf`, which matches the word **INFO**;
  the six "extra" lines were ordinary INFO records from running an additional momentum phase. Re-run
  against WARNING/ERROR lines only: **0 distress lines on all 10 runs, both arms.** **G18.2 passes
  cleanly** and the earlier concern is withdrawn.
  🔑 **V3 — the load-bearing check, and it needed care to read.** Batch 21's `O1/shipped` figures were
  computed offline *at the same state*; a matched-`t` comparison of two **different trajectories**
  measures something else, and taken raw it looks like a contradiction (confined branch reads
  0.977–0.991, i.e. the arm driving **less**, where the prediction was 1.001–1.005, i.e. **more**).
  Resolved by measurement, not argument: on the confined branch the arm's `R2` is **larger**
  (1.0047–1.0085) and its `Pb` correspondingly **lower** (0.9764–0.9871) — a bigger bubble at lower
  pressure. Dividing the state divergence out leaves the scheme's own effect:

| | predicted offline (Batch 21) | measured on the arm's own trajectory |
|---|---|---|
| B3M energy | 1.001 | **1.0015** |
| B3M implicit | 1.005 | **1.0052** |
| B3M momentum | 0.494 | **0.5391** |
| B3MW01 momentum | 0.325 | **0.3241** |
| B3MW01 transition | 0.340 | **0.3085** |

  **The offline screening predicted the arm's own behaviour on both branches**, the confined side to
  four significant figures once the trajectory divergence is factored out. That is the strongest
  evidence in this workstream that the screens model what the code does. It also explains why a
  +0.5% instantaneous excess does not run away: the response is **negatively fed back** — more drive
  → bigger bubble → lower `Pb` → less drive.
  ✅ **V5 / G18.4 — the branch discontinuity is genuinely gone, and a THIRD jump figure appears.**
  The baseline has **exactly one** `P_HII` 0→positive crossing per driving config (B3M at
  `t` = 0.2980, B3MW01 at 0.7146) with an **observed** adjacent-snapshot drive step of **+6.79%** and
  **+5.97%**. The arm has `P_HII` > 0 on **every row of every config** and therefore **zero
  crossings**. ⚠️ Note this is a *third* distinct number for "the jump", and all three are correct
  answers to different questions: **+23.4%** the analytic discontinuity at the exact crossing;
  **+34%** the branch's effect evaluated at a fixed post-crossing state; **+6.79%/+5.97%** what the
  integrator actually experienced, i.e. the branch jump net of the `Pb` decline between snapshots.
  The last is the operationally real kink and is the one to quote for trajectory effects.
  ⚠️ **What O1 does NOT improve: the phase seams.** Adjacent-row `P_drive` steps across
  `energy>implicit`, `implicit>transition` and `transition>momentum` are **essentially identical**
  between arms (B3M 17.60/17.63, 15.91/15.62, 0.52/0.47; B3MW01 17.18/17.09, 11.07/9.35, 0.15/0.29;
  F1HI 17.44/17.57, 91.94/91.38). Those steps are driven by the phase machinery — `Eb`
  re-initialisation and the `max(P_thermal, P_ram)` handover — not by `P_HII`, so removing the
  `P_HII` branch cannot and does not touch them. **The continuity claim for O1 is about the branch
  switch only**, and should be stated that way rather than as smoothness in general.
  **G18.5 is MOOT under O1**, not owed: it asked whether the closure's own dust model bites in the ED
  phases, and O1 does not carry a closure dust model — it reads `R_IF` from the shell solve, which
  already integrates dust. Recorded closed rather than outstanding.
  ⚠️ **Artifacts:** `harness/b21_arm_verify.py` and `data/b21_arm_verify.csv` are written but
  **NOT committed** — the uncommitted `.gitignore` change ignores `docs/dev`, so new files there need
  `git add -f`. The 40 MB of raw run outputs under `data/hpc/` are likewise untracked, which is
  correct: the 💾 rule wants the derived CSV committed, not the run dirs.

- **2026-08-29 (two corrections to my own standing claims, from the arm data)** —
  ✏️ **(1) "Coverage is 2 configs of one cloud" is STALE and I kept repeating it after it stopped
  being true.** That was correct for the *offline* screening (B3M + B3MW01, one cloud, two wind
  strengths). The **arm ladder ran five configs spanning four distinct clouds**, and the axes Batch
  20 slice 4 flagged as untouched are now largely covered:

| | span covered by the arm ladder |
|---|---|
| `nCore` | 2.94e57 → 2.94e61 — **4 decades** |
| `mCloud` | 7e4 → 9.9e6 — **2.2 decades** |
| `sfe` | 0.01 → 0.5 — **the full registered range** |

  What remains genuinely untested: **PRB** (the compact probe), **WW** (weak wind / collapse), the
  non-power-law profiles (`BE`, `PL2`), the other wind rungs, and late times (`B3ML`). So the
  coverage blocker should be restated as **"the compact and collapse corners are untested"**, not as
  "two configs".
  ⚠️ **(2) The freeze ratchet is STRUCTURAL under O1, not incidental — and harder to fix than I
  implied.** Measured on the arm's own output: `P_drive == P_HII` on **481 of 481** confined-branch
  rows (SC, F1LO, B3M × energy+implicit), and `P_drive == Pb` on **zero**. That is by construction —
  O1 returns `P_conf·ρ` with `ρ ≥ 1`, so it always wins `max(press_bubble, P_HII)`. **Every
  confined-branch step is therefore driven by a value frozen at the start of its ODE segment.**
  Two consequences I had not stated. **(a)** The ratchet's effect is **already inside the measured
  ΔR2** for these five configs — it is not a hidden risk there; the runs were clean (0 distress) and
  the trajectories are smooth and predictable. The open worry is the **compact** regime, where §3's
  C2a row rates this class *"catastrophic at compact scale"* and **PRB has never been run**.
  **(b) The obvious fix does not work for O1.** Calling the helper live inside the ODE RHS would
  require a **shell solve per ODE evaluation**, because O1 reads `shell_props.R_IF` — prohibitive.
  So the options are to interpolate `R_IF` across the segment, to bound the freeze error and accept
  it, or to accept a scheme whose confined branch is piecewise-constant by construction. **This is a
  design decision, not a bug fix**, and it is the last substantive thing between O1 and a
  recommendation.

- **2026-08-30 (trigger/fate/P_drive census lands — `trigger-fate-pdrive-audit.md`)** — On the
  maintainer's three questions (are the transition/momentum/collapse triggers physically sound;
  is the bubble fate reflected; does `P_drive` make sense, incl. the b14 `R_IF/R2` scheme), a
  full census of the working tree at `b8f77276` was gathered by two independent read-only source
  passes and written up as `docs/dev/phii-identity/trigger-fate-pdrive-audit.md`. Headlines, in
  this workstream's terms: **(1)** every *phase-advance* trigger (velocity-sign, cooling_balance,
  energy_to_momentum, no-root streak, energy_floor, ram_dominated) is recorded **nowhere** in
  `metadata.json` — `main.py` discards the runners' `termination_reason`; only the log-scraped
  `transition_channel` survives. **(2)** 1b/1c solver deaths hand off silently instead of ending
  the run; `ev:694`'s substring test sets `isCollapse=True` on `large_radius_event`; the collapse
  fate has two radii (event `1.5·coll_r` vs inline `coll_r`). **(3)** the P_drive findings here
  (freeze ratchet both sides, two-Strömgren-volumes gate/value split, un-ramped predicate,
  momentum factor-~2 branch jump) were re-verified independently and none contradicts the ledger;
  K10-O1 applies cleanly at `b8f77276` (`get_phii_c3c` byte-identical to the patch base, +26-line
  offset from `mass_freeWind`). New, unmeasured observation for any O1 gate: the EOM debits
  `P_ext` at `R2` while O1 amplifies the drive to the `R_IF` surface — negligible while
  `P_ext ≪ P_drive`, unexamined near stall. Maintainer-decision queue is §6 of the audit doc;
  it adds nothing to D5 but sequences the segment-freeze call first.

- **2026-08-30 (PRB + WW arm ladder — ⛔ O1 FAILS ON THE COMPACT CONFIG. The §3 prediction lands.)**
  — Sweep `phii_arm_20260829_144118Z`, `CONFIGS=PRB,WW`, both arms at `cce8c924`. The config
  override reached the job this time (only PRB and WW ran), so the positional-argument fix works.

| config | baseline | k10_o1 | slowdown |
|---|---|---|---|
| **PRB** | **ok**, 672 s, 234 snaps, `energy>implicit>transition>momentum`, `shell_collapsed` | ⛔ **timeout > 7200 s**, 210 snaps, **only `energy>implicit`** | **> 10.7×** |
| WW | ok, 533 s, 164 snaps | ok, 574 s, 173 snaps | 1.1× |

  ⛔ **PRB does not complete under O1.** The baseline finishes in 11 minutes and reaches all four
  phases; the arm was **killed at the 2-hour wall limit having never left the implicit phase**.
  ⚠️ **The ledger's `FATE-CHANGE shell_collapsed → NA` must NOT be read as a fate change.** `NA` is
  the absence of a terminal state because the run was *killed*, and the `+44.5%` ΔR2 compares a
  completed baseline against a trajectory frozen mid-implicit. By this workstream's standing rule —
  *"a run that never reaches the phase a gate needs is VOID, never a confirming null"* — **the PRB
  comparison is VOID as a fate/ΔR2 measurement**. What it *is* is a hard performance/stiffness
  failure, and that is the finding.
  ✅ **WW passes**: 1.1× wall time, no fate change (`shell_collapsed → shell_collapsed`), OVER-BAR at
  **+14.15%** — the same magnitude class as the other confined-branch configs (SC +5.28%, F1LO
  +2.08%), and the same sign, since WW never reaches the driving branch either.
  🔑 **This is §3's C2a prediction, arriving where it said it would.** That row rates the
  per-segment freeze ratchet *"**catastrophic at compact scale**"* (citing phase1a-init Extra finding
  #1), and **PRB is the compact probe** — the one config the ladder had never covered. O1 activates
  that ratchet on **100% of confined rows** (`P_drive == P_HII` on 481/481, measured 2026-08-29), and
  PRB is precisely where the class was predicted to bite. The slowdown is *not* per-call cost — under
  O1 the helper only reads `shell_props.R_IF` and is cheap — so it is **more ODE steps**, i.e.
  stiffness, which is the signature a piecewise-constant drive would produce.
  ⚠️ **Stated as the strong hypothesis it is, not as a conclusion.** The freeze ratchet is the
  predicted and most likely cause, but it is **not yet demonstrated** for this run: diagnosing it
  needs PRB's own `dictionary.jsonl` and `trinity.log` (step sizes, where it stalls in implicit),
  which stayed on `/gpfs` — `./sync.sh down-all` for stamp `20260829_144118Z`. Competing
  explanations not yet excluded: PRB-specific stiffness independent of the freeze, or an interaction
  with the `dt_switchon` ramp at compact radii.
  **Consequence for the recommendation, stated plainly: O1 cannot be adopted as it stands.** Its
  record is otherwise strong — five configs over four decades of density, zero distress, no fate
  changes, offline screens predicting reality to four significant figures — but it **fails to
  complete a registered core config**, and that is disqualifying regardless of how good the rest
  looks. The freeze is structural under O1 (it wins the `max` by construction) and the obvious fix
  is barred (a live call needs a shell solve per ODE evaluation), so this is now the **central**
  design problem, not one of three blockers. My 2026-08-29 line that "the only thing that could
  still overturn this is PRB" is hereby cashed: it did.

- **2026-08-30 (PRB diagnosed from the raw run — my freeze-ratchet hypothesis is REFUTED, and the
  real finding is bigger: O1 UNBINDS the compact config)** — Pulled PRB's own snapshots rather than
  reasoning from the wall time.
  ⛔ **The stiffness/ratchet explanation I recorded yesterday is not supported.** `P_HII` repeats
  exactly on **0 of 78** adjacent implicit rows in the arm — no staircase is visible at snapshot
  resolution (the freeze is *within* a `solve_ivp` segment, i.e. between snapshots, so snapshot data
  cannot see it either way). It is **not evidence for** the ratchet; it is the absence of the
  evidence I claimed. Withdrawn as the explanation.
  🔑 **What actually happens: the two arms are on qualitatively different trajectories.**

| | baseline (C3c) | k10_o1 |
|---|---|---|
| t = 0.055 / 0.017 | R2 0.331, v2 **+3.30** | R2 0.182, v2 **+6.06** |
| t ≈ 0.30 | R2 0.779, **momentum**, v2 **+1.10** | R2 **1.134**, **implicit**, v2 **+4.67** |
| t = 0.443 | R2 0.864, v2 **−0.025** ← turning around | — |
| end | R2 **0.01000** (the collapse floor), v2 −11.5, `shell_collapsed` | still expanding when killed |

  **The baseline decelerates, turns over and collapses to the 0.01 pc floor. The arm never turns
  over** — `v2` is +4.67 and *rising* at the point it was killed. O1's confined-branch excess is
  enough to **unbind a shell that C3c collapses**.
  🔑 **So the timeout is a SYMPTOM, not the disease.** Collapse terminates a run early; without it
  the arm must integrate the full `stop_t`, and it does so entirely inside the **implicit** phase,
  which is the expensive one (the βδ solver calls `get_bubbleproperties_pure`). The baseline spends
  most of its wall time in the cheap momentum phase. That, plus ~3× smaller steps (median snapshot
  `dt` 2.32e-5 vs 7.27e-5), accounts for the 10.7×.
  ⚠️ **This is the first substantive FATE CHANGE in the whole ladder** — `shell_collapsed` → not
  collapsing — and it is a **D3-level finding**, far more consequential than a performance problem.
  D3 ruled that a *timing* change under an explained mechanism is acceptable but that **fate flips
  remain reportable**. This is a flip, on a **registered core config**.
  🔑 **And it sharpens what the confined-branch excess means.** That excess measures **+0.475%** of
  `P_conf` (G21.4) and moves `R2` by +2–5% on SC/F1LO/WW. On PRB — compact, marginal, and sitting
  near the collapse boundary — **the same ~0.5% flips the outcome**. So the excess is not merely
  "not negligible" (2026-08-29's correction); on a marginal config it is **decisive**. Anything that
  adopts O1 must confront that a sub-percent drive change reverses a fate.
  **Open, and needed before any verdict:** whether PRB *should* collapse is a physics question this
  ladder cannot settle — C3c says yes, O1 says no, and neither is self-evidently right. The
  cheapest next measurement is to re-run PRB under the arm with a **longer wall limit** (the 2 h cap
  was mine, not physics) to see what fate it actually reaches; `--time` in `b14.sbatch`. Until then
  PRB's arm fate is **unknown**, not "no collapse".
