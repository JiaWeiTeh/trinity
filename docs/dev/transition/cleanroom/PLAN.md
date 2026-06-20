# Clean-room redo: implicit→momentum transition trigger — certify substrate, then characterize

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
> exists, revise the plan and note what changed and why (date it). Leave the plan
> better than you found it. **Keep all banner paragraphs at the top of every
> plan and analysis doc.**
>
> 💾 **Persist diagnostics — commit, don't re-run.** The container is ephemeral
> and full/hybr runs cost hours, so any diagnostic worth keeping must be saved as
> a committed artifact (a CSV/table under `docs/dev/data/`, or a force-added
> harness/figure in the relevant `docs/dev/<workstream>/` folder as the hybr work did) — never left in `/tmp` or
> an untracked `outputs/`. A future visit must be able to reproduce or compare
> against the numbers **without re-running**; record the exact config + command
> that produced each artifact.

**About this document**
- **Status (2026-06-20):** 🟡 **IN PROGRESS** — pre-registration. C0 substrate-certification gate defined; C0.3 audit done; C0.2 harness built (smoke-validated); C0.1 staged. No production change. No candidate trigger evaluated yet.
- **Type:** plan + pre-registration — a *clean-room* redo of the implicit→momentum transition-trigger investigation that deliberately does **not** inherit prior numerical results/rankings/verdicts (see §1).
- **Workstream:** `transition/` (clean-room subdir `cleanroom/`).
- **Where it sits:** supersedes-as-entry-point the **quarantined** `docs/dev/transition/{TRIGGER_PLAN,P0,pshadow-design}.md` and the background `docs/dev/archive/betadelta/*`. Those remain on disk as prior art; their *conclusions* are quarantined (§1), their *methodology and candidate menu* are reused.
- **Code it concerns:** the implicit-phase terminator `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1095` (the `(Lgain−Lloss)/Lgain < threshold` trigger); the betadelta solver `trinity/phase1b_energy_implicit/get_betadelta.py`; bubble cooling `trinity/bubble_structure/bubble_luminosity.py`; the transition phase `trinity/phase1c_transition/run_transition_phase.py`.
- **Linked files & data:** harness `docs/dev/transition/cleanroom/c0_consistency.py`; data `docs/dev/transition/cleanroom/data/`.

---

## 0. Why a clean-room redo, and the trust boundary

**Problem.** Under the default `betadelta_solver=hybr`, full runs **stall in the
implicit energy phase and never reach momentum** — the cooling-balance trigger
`(Lgain−Lloss)/Lgain < 0.05` (`run_energy_implicit_phase.py:1095`) plateaus well
above 0.05 and never fires, so runs sit in implicit until the `stop_t` cap. This
is a real behaviour the more-correct hybr solver exposed (legacy clamped β∈[0,1],
which could only ever let Pb decline → monotone approach to cooling balance).

**Why clean-room.** Prior efforts on this exist across branches/docs. Their
*ideas* are valuable; their *numbers, rankings, and verdicts* may be contaminated
(scratch `/tmp` captures, pre-hybr code, stale line refs). To produce results we
can trust and publish, we **re-blind, re-derive, regenerate**: keep the rigor
(shadow diagnostics, pre-registered gates, committed artifacts), discard the
verdicts.

**Trust boundary (maintainer decision, 2026-06-20): _certify, then build_.** We do
not assume the substrate (the hybr solver + the `Lgain`/`Lloss`/`Pb`/`Eb`/β/δ
machinery the trigger reads) is correct. We **certify it with a cheap, independent
gate (C0, §2) first.** If C0 passes, the substrate is trusted ground and the
investigation is scoped to the trigger. If C0 fails, the contamination reaches the
substrate and scope escalates. **Production stays frozen** (default
`instantaneous` trigger unchanged) — all work here is offline harness + docs +
committed CSVs; nothing under `trinity/` changes.

---

## 1. Contamination policy — ideas in, results out

**Reusable (ideas, cheap, falsifiable — we test them anyway):** the candidate
trigger menu (§5), the physical hypotheses (§5, stated as *open* hypotheses), the
methodology (shadow/pre-registered gates), and the config span (configs are
inputs, not results).

**Quarantined (do NOT inherit; provenance-unknown until regenerated from the
pinned SHA):** every firing-epoch number, ranking, ε-sensitivity result, β/δ
range, "gate passed/failed" verdict, and figure from prior work. Do not cite or
build on any of them.

> **Disclosure (2026-06-20):** during idea-extraction, a subagent leaked a few
> prior *verdicts* (e.g. a per-config retained-energy value, an "F1 retreats under
> surge" claim, a "clock-B is not a macro driver" claim). **These are explicitly
> quarantined and must not seed any gate or expectation.** They are listed here
> only so a reader knows to disregard them. That they slipped through is exactly
> why C0/G0 are pre-registered against an *independent oracle* (§3), not against
> prior conclusions.

**Quarantine inventory (paths only; do not read for their numbers/verdicts):**
- Docs: `docs/dev/transition/P0.md` (all results/verdict sections), `docs/dev/transition/TRIGGER_PLAN.md` (firing claims), `docs/dev/transition/pshadow-design.md` (evidence section); `docs/dev/archive/betadelta/{PHASE2_ARMS,stalling-energy-phase,HYBR_PLAN,PHASE0_BASELINES}.md` (results/verdict sections).
- Prior harness (do not reuse/extend): `docs/dev/transition/harness/` (`harvest.py`, `psens.py`, `heartbeat.py`, `*.param`).
- Data CSV/JSONL: `docs/dev/data/transition_*.csv`, `docs/dev/data/hunt_h*.csv`, `docs/dev/data/stalling_*.csv`, `docs/dev/archive/betadelta/diagnostics/*.{jsonl,csv,csv.gz}`.
- Figures: `docs/dev/archive/betadelta/diagnostics/*.png`, `*.gif`.

Methodology/scope sections of the quarantined docs (the "how", not the "what we
found") are safe to reuse.

---

## 2. C0 — substrate certification gate (PRE-REGISTERED)

Certify the substrate with checks that are independent of the contested trigger
and, where possible, of the numerics. **Bars are pre-registered here, before any
run** (adjustable only with a dated note explaining why, per the 🔄 banner).

### C0.3 — `Lloss` is pure radiative (code audit) — ✅ DONE (2026-06-20)
Verified against current source: `bubble_LTotal = L_bubble + L_conduction +
L_intermediate` (`bubble_luminosity.py:790`), each a radiative integral —
`χ_e·n²·Λ(T)·4πr²` (CIE, `:696`) or `dudt(n,T,φ)·4πr²` (non-CIE, `:743/780`).
**No PdV term, no velocity** in any integrand (`v_array` is computed but never
enters the luminosity). ⇒ the ratio's `Lloss` is a clean cooling-vs-injection
fraction; PdV is carried separately in the energy ODE (next).

### C0.2 — β/δ ↔ trajectory consistency (internal, code-derived) — harness built
**Code-grounded reframing (important).** The energy ODE is **not** a naive
`dEb/dt = Lgain − Lloss − PdV` budget; it is the Rahner-thesis A12 equation
(`get_betadelta.py:182` `cool_beta_to_Ebdot_pure`), where cooling enters
*implicitly via β*, not as an explicit `Lloss` term. So a naive energy-budget
closure would test a relation the code doesn't enforce. The relations the code
*does* enforce by definition are:

- `β ≡ −(t/Pb)(dPb/dt)`  ⇒ predicted `dPb/dt = −β·Pb/t`  (`get_betadelta.py:248`)
- `δ ≡ (t/T)(dT/dt)`     ⇒ predicted `dT0/dt = δ·T0/t`   (`get_betadelta.py:294`)

**Check:** finite-difference the stored `Pb(t)`, `T0(t)` across consecutive
implicit-phase snapshots and compare to the predictions from the stored β, δ.
This certifies the solver's (β,δ) outputs are consistent with the integrated
trajectory — independent of the trigger.

**Pre-registered bars:** (i) median relative residual ≤ **5%** over implicit-phase
rows for *both* `dPb/dt` and `dT0/dt`; (ii) the median residual **shrinks under
timestep refinement** (first-order finite-difference truncation → ∝ Δt), i.e. it
is consistency error, not a systematic offset. Fail ⇒ the β/δ the trigger's
denominator/cooling depend on are not trajectory-consistent ⇒ escalate scope.

> **Smoke observation (2026-06-20, harness mechanics only — provenance-unknown
> legacy run `outputs/mockOutput/mockFullrun`, 49 implicit rows; NOT a
> certification):** `β↔dPb/dt` median **2.2%**, p90 8.7% — a *genuine* cross-check
> (`Pb` from the structure solve / `bubble_E2P(Eb,…)`, `β` from the solver, so
> agreement is non-trivial). `δ↔dT0/dt` came out **exactly 0.000%**, which is a
> **red flag, not a pass**: it strongly suggests `T0` is advanced by the *same*
> δ-ODE (`delta2dTdt_pure`, `dT/dt=(T/t)δ`), making the δ check **tautological**
> (certifies nothing). **Open item before C0.2 counts:** confirm whether the δ
> check is independent; if tautological, drop it and rest C0.2 on `β↔Pb` (genuine)
> plus the external C0.1. Do not report the δ residual as evidence until resolved.

### C0.1 — analytic adiabatic Weaver null (external truth) — STAGED
The strongest, numerics-independent check: in the adiabatic limit (`Lloss→0`) the
energy-driven bubble has the closed-form Weaver (1977) solution
(`R₂ ∝ (Lt³/ρ)^{1/5}`, `Eb = (5/11)L_w t` — cf. `get_InitPhaseParam.py:166`; a
constant β; `T` from Weaver Eq. 37).

**Blocker + resolution.** There is **no cooling-off parameter** (registry has only
`cool_alpha/beta/delta` + table paths; `net_coolingcurve.get_dudt` has no disable
path). Rather than add a production flag (would break "frozen"), the harness will
**monkeypatch the cooling lookups to zero inside the offline harness only**
(`bubble_luminosity.py:694` CIE `Lambda_bubble`, `:741/779` non-CIE `dudt`) — the
same runtime-wrap pattern the prior `velstruct/hunt.py` used for the solver.
Production source stays byte-identical (nothing in `trinity/` changes).

**Pre-registered bars:** (i) code `R₂(t),Eb(t),Pb(t),T(t)` match analytic Weaver to
**≤2%** over the energy/implicit phase; (ii) **hard gate** — with `Lloss≡0` the
ratio `(Lgain−Lloss)/Lgain → 1` and the transition trigger **never fires** (any
firing ⇒ mis-normalized substrate, STOP).

**Feasibility risk + fallback.** If the betadelta solver cannot converge with
`Lloss≡0`, fall back to **C0.1b** (below). The hard "never-fires" gate still holds
regardless of convergence (it tests the trigger, not the structure).

### C0.1b — Weaver early-phase scaling (external, no patch) — STAGED, fallback
With cooling on but sub-dominant early, `R₂(t)` in the energy phase should track
the analytic power law `R ∝ t^{3/(5−|α|)}` (cf. `paper/methods/figures/paper_radiusComparison.py`).
Weaker than C0.1 (cooling contaminates; tests phase 1a, a different path) but
needs no patch. Pre-registered bar: power-law exponent recovered to ≤5% over the
early energy phase.

**C0 verdict rule:** substrate is **certified** iff C0.3 ✅ (done) **and** C0.2
passes **and** (C0.1 passes, or C0.1 infeasible **and** C0.1b passes **and** the
C0.1 hard never-fires gate passes). Otherwise scope escalates to substrate
re-verification.

---

## 3. The independent oracle (for the later trigger harvest, not C0)

Every candidate trigger is judged against a reference that depends on **no
candidate and no threshold**: the **PdV-inclusive net-energy zero crossing** — the
first time the bubble stops gaining energy,
`Lgain − Lloss − 4πR₂²·v₂·Pb ≤ 0` (the `Eb`-peak). A good trigger fires at/just
after this peak. Cross-checks: the adiabatic null (§2, must never fire) and the
retained-energy fraction `Eb/∫Lgain dt` at firing (literature sanity band — verify
the range before citing). The oracle is **diagnostic only**; it never drives a run.

---

## 4. Config span (inputs; reused, not inherited results)

The six characterized configs (already pass GMC plausibility), spanning feedback
strength × cloud density × profile, plus one strong-WR/SN-jump config:

| config | mCloud | sfe | nCore | rCore | profile | note |
|---|---|---|---|---|---|---|
| large_diffuse_lowsfe | 1e7 | 0.01 | 1e2 | 1 | densPL α=0 | weak feedback, diffuse |
| simple_cluster | 1e5 | 0.3 | 1e5* | 0.01* | densPL α=0 | baseline |
| small_dense_highsfe | 1e4 | 0.5 | 1e6 | 0.1 | densPL α=0 | strong, dense |
| midrange_pl0 | 1e6 | 0.1 | 1e4 | 0.01* | densPL α=0 | mid |
| **pl2_steep** | 1e6 | 0.1 | 1e5 | 1 | densPL **α=−2** | **crux** (steep halo) |
| be_sphere | 1e6 | 0.05 | 1e4 | 1 | densBE Ω=14 | BE profile |

`*` = schema default. The steep config is the **crux**, not an afterthought: for
r⁻² `Lloss ∝ n²` may collapse before any cooling family fires (§5).

---

## 5. Candidate menu + hypotheses (IDEAS — unverified, to be tested)

**Candidate trigger families** (definitions only; *which fires when* is to be
measured, not inherited):
| id | family | criterion |
|---|---|---|
| F0 | instantaneous rate-ratio (**current**) | `(Lgain−Lloss)/Lgain < ε` |
| F1 | cumulative energy | `∫Lloss dt / ∫Lgain dt > 1−η` |
| F2 | timescale | `t_cool/t_dyn < k` (define `t_cool`,`t_dyn` explicitly; report sensitivity) |
| F3 | force / continuity | `4πR²Pb / (surviving forces) < O(1)` |
| F4 | blowout (geometric) | `R2 > rCloud` |
| F5 | mixing-flux balance | (no sharp 1D transition; cite, do not model) |

**Open hypotheses (to confirm/refute from regenerated data — NOT findings):**
1. The trigger's **instantaneous numerator** resets the ratio *upward* on each
   feedback episode (WR/SN switch-on spikes `Lgain`), so a cooling event can be
   masked. Integrating over episodes (F1) or a sustained-over-`t_cross` rule may
   remove it. *(hypothesis)*
2. For a **steep r⁻² halo**, `Lloss ∝ n²` collapses as the bubble expands, so no
   cooling family (F0–F2) may ever fire; the physical fate is **blowout (F4)** /
   force-subdominance (F3). ⇒ the right trigger may be **profile-dependent**, which
   a single scalar cannot express. *(hypothesis — the crux to settle)*
3. The implicit-phase duration ("clock A", reach-transition) and the 1c
   transition-phase length ("clock B", sound-crossing drain) are **distinct
   clocks**; separate before explaining any "long transition". *(hypothesis)*

---

## 6. Downstream (sketched; detail deferred until C0 passes)
- **H0 — harvest (shadow/offline):** from the pinned SHA, per implicit-phase
  segment, log each family's would-fire epoch **and** the §3 oracle; commit CSVs +
  harness + exact command.
- **G0 — divergence gate (pre-registered):** which families track the `Eb`-peak
  *without resetting* across the WR/SN jump, across the full span incl. the steep
  crux. "No single scalar works → profile-dependent" is an allowed, publishable
  verdict.
- **Shadow → validate → default decision**, all behind a `transition_trigger`
  param with **default unchanged** (`instantaneous`).

---

## 7. Reproduce / artifacts
- Pinned baseline SHA: recorded in `data/` outputs (see harness `--meta`).
- C0.2: `python docs/dev/transition/cleanroom/c0_consistency.py <param-or-jsonl> [--stop-t T] [--out CSV]`.
- Data lands in `docs/dev/transition/cleanroom/data/` (committed).
