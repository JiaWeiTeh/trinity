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

**Status (2026-08-13):** 🔵 actionable — **Batches 0, 1, 3, 4a and 5-stage-1 done. D1/D2 answered.
An independent adversarial audit (2026-08-13) corrected 2 critical + 11 major items — see §9, and do
not quote any figure from an unmarked earlier revision of this file.**

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
- **Next:** Batch 5 stage 2 — run the C3c arm (B2M/B3M/WW + PRB/B1M controls), fates enumerated
  under D3. Expect >5% ΔR2 by construction; the arm decides whether fates survive.

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
| momentum | `P_HII + P_ram = 2·P_ram` | `P_C3a + P_ram` |

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

**Stage 2 (not started).** Run **C3c** as an arm on B2M, B3M, WW (momentum-reaching) plus PRB (must
be near-inert: only its energy-phase ramp window changes) and B1M, gated exactly as Batch 3 was —
matched-`t` ledger, fates enumerated under D3. Expect ΔR2 well over the 5% bar by construction
(momentum drive ×2.4–4.3, early-energy drive ÷ up to 3.3); the questions the arm answers are whether
fates flip, whether the integrator tolerates the transition handover kink, and what the fate map
looks like vs stock. Implementation note for the arm: replace the `n_IF_Str` → `P_HII` path with the
§3c branch — smallest diff is computing `P_C3a` in the phase runners next to the existing `P_HII`
lines and selecting per §3c's table; the cap and shell structure stay untouched (they still serve
absorption fractions and diagnostics).

### Batch 6 — land — Status: ⬜
Chosen candidate (D1 decides between C1, C1⊕C2b, or a C3) on the **full-12**; full ladder
re-verify; CHANGELOG entry; reconcile the evidence README (§7 answers), DOC_STATUS, and — when
the sibling branches merge — fold-back notes for momentum-pdrive (its §2 "inferred" caveat, its
CSV column rename) and weak-winds (quantitative collapse times now clean). Goldens re-baselined
under D4 with a table of before/after.

## 7. Decisions needed from the maintainer

| id | question | blocks | state |
|---|---|---|---|
| D1 | ✅ **ANSWERED 2026-08-12/13** — the momentum sum **is** intended, conditional on `P_HII` being genuinely its own calculation; the transition `max` is a **deliberate** smooth handover as `Pb → 0`. See §2. Open remainder: whether a better handover formulation exists, and what C1 actually costs (Batch 3, unrun at the time of writing) | Batch 3 verdict | **answered; C1 still unmeasured** |
| D2 | ✅ **ANSWERED 2026-08-12** — `P_HII` should be a real, separate pressure, treated as one unless the architecture cannot support it (then the assumption must be explicit). Consequence: the target is **decoupling**, and §3b shows the cap is not the coupling — the ionised volume is. Open sub-question for Batch 5: which decoupled formulation (C3a/C3b/C3c) | Batch 5 | **answered; formulation open** |
| ~~D2-old~~ | ⛔ superseded by the above. **WAS THE CRUX (Batch 4a).** Removal is proven *safe* — no blow-up materialises in any regime tested, including the compact probe. So the question is no longer "can we?" but "should we?": is the uncapped Strömgren pressure physically trustworthy at these ionized volumes, given it exceeds `Pb` on 100% of rows (up to 7.79×; the 3.36 quoted earlier was PRB's `blowup_max`, not the matrix max) and shifts trajectories 15–28%? No measurement can settle this; it needs the model's intent. Also confirm §2's reading that the cap was pragmatic, not a physics claim. | Batch 4b design; Batch 5; **4a landing** | **open** |
| D3 | Fate flips under a candidate fix: acceptable-if-explained, or a re-tune trigger? | Batch 3/4 verdicts | **open** |
| D4 | Authority to re-baseline goldens (`test_phase_boundary.py`, `test_betadelta_hybr_stress.py`, `test_scheme_screen.py` fixtures) if the landed fix moves them. | Batch 6 | **open** |

## 8. Ledger (results land here — the one source of truth)

### 8.1 Batch verdicts
| batch | status | date | verdict (one line) | artifacts |
|---|---|---|---|---|
| 0 | ✅ | 2026-08-12 | **PASS** on 6/6 core. Identity holds on 100% of implicit and transition rows and 26/27 momentum rows (WW's final collapse row is stale-`Pb`) and ≥96.97% of energy rows, relΔ ≤2.9e-16, across 4 decades of nCore. B3M independently reproduces momentum-pdrive (`P_HII` vs `P_ram` = 2.39e-16 over 34 rows). Drive anatomy: implicit exactly 1, transition ≤1.998 (median 1.82), momentum exactly 2.000, energy ≤3.31. **`frac_nIFStr_eq_n0` = 1.0000 in every phase of every config** — the cap is bound everywhere, needing no diagnostic to show it | `data/b0_identity_grid.csv`, `data/b0_trajectories.csv`, `data/b0_walltimes.csv` |
| 1 | ✅ | 2026-08-12 | **PASS.** G1(i): B3M 231 + PRB 184 + WW 178 = **593 rows** exactly equal on every pre-existing key (repr compare), matching row counts ⇒ diagnostic inert; independently corroborated by the matched-t comparator returning 0.000% on both. Cap binds **100% of rows in every phase**; blow-up p99 1.06–7.79, max **7.786** (WW momentum; B3M 3.331, PRB 3.306, B1M 3.308). **Kill bar NOT tripped ⇒ C2a survives, Batch 4a authorised.** Corrects B0: sub-100% energy rows are `Pb` staleness at the 1a→1b handoff, not cap-slack | `data/b1_bitidentity.csv`, `data/b1_capmap.csv` |
| 2 | ⬜ | — | — | — |
| 3 | ✅ | 2026-08-13 | **C1 MEASURED — safe, small, and aimed at the wrong target.** Momentum-only `max(P_HII, P_ram)` (halving `P_drive` from `2·P_ram` to `P_ram` there) on 4 configs spanning weak winds, two masses and two bench radii. **All WITHIN-BAR, no fate changes:** B1M **0.000%**, B2M 1.24%, B3M 4.00%, WW 1.29% ΔR2 at matched `t`. B1M is the pre-registered falsifiable control — it never reaches momentum, so C1 must be inert there, and it is to 0.000%. The effect is small because momentum is only 12–15% of these runs. **Verdict: C1 does not break anything, but it does not do what D2 asks** — with `P_HII ≡ P_ram` in momentum, `max(P_HII, P_ram) = P_ram`, so C1 *deletes* the photoionised channel rather than decoupling it, and D1 says the sum is intended. Superseded as a fix by C3; retained as the measured cost of the double-count | `data/b3_c1_ledger.csv` |
| 4 | 🟡 | 2026-08-12 | **4a MEASURED — survives, but is not behaviourally neutral.** 4/4 configs (PRB, B3M, F1HI, F1LO) ran to their natural end, **zero** distress lines (no excess-work, overflow, monotonic-guard or convergence warnings), wall times *within* baseline (492–764 s vs 682–832 s). **No fate changed** on any config. Identity destroyed as intended: `frac_PHII_eq_Pb` = **0.0000** in every phase of every config (was ≥0.9697), relΔ now O(1) (0.06–2.55). But **every config breaches the 5% bar**: ΔR2 max 15.3–28.4%, all located inside the `dt_switchon` window (t = 1.3e-7 … 9e-6 Myr); ΔR2 at end-of-overlap 0.95% (PRB, recovers) → 14.4% (F1LO, retained). Mechanism: uncapped `P_HII` exceeds `Pb` on **100%** of rows (max 7.79× across the matrix; 3.36× on PRB) so it wins the `max`, lifting median `P_drive/Pb` from 1.0000 to 1.83 (PRB). **Verdict: C2a is numerically viable and physically consequential — not a free win. Landing it needs D2.** 4b not started | `data/b4a_ledger.csv`, `data/b4a_identity_grid.csv` |
| 5 | 🟡 | 2026-08-13 | **Stage 1 (offline screen) done — C3b ⛔ REJECTED, C3a advances.** No solver run: both candidates are closed-form in stored quantities, evaluated on the stock trajectory across 5 configs. C3b fails the pre-registered wind-only limit *structurally* — `n = n_cloud(R2)` has **no `Qi` dependence**, so switching the ionizing source off leaves its `P_HII` unchanged; it also steps 4 decades at `rCloud`. C3a is causally decoupled (`Qi`, `R2` only), has the correct `Qi → 0` limit, and gives sensible ionised densities (19–8055 cm⁻³ in momentum) — but sits uniformly **3.5–7.6× above `P_ram`** and never crosses it, i.e. predicts a photoionisation-dominated momentum phase in all five configs. **Stage 1b: C3c designed (§3c) and screened — it supersedes bare C3a.** The confined skin has no independent density (any decoupled-thickness skin is C3a × O(1), *higher*), so C3c is a regime switch: transmit when `P_C3a ≤ P_conf`, drive at `P_C3a` when above. Screened on the same 5 runs: implicit **exactly** untouched (ratio 1..1..1), D-ramp fixed as a side effect (energy ratio down to 0.30 = the ramp honoured), `t_cross` inside transition in all 4 configs that reach it, momentum drive 2.4–4.3× stock. Stage 2 (run arm) not started | `data/b5_c3_screen.csv`, `data/b5_c3c_regime.csv` |
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
