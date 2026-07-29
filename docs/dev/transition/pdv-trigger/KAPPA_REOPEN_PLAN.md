# Re-opening `cooling_boost_kappa` — the K0 re-read and the K1–K4 HPC campaign

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

**Status (2026-07-29):** 🔵 actionable — **K0 (the offline re-read) is DONE** and is the evidence base below
(`data/kappa_eq47_check.csv`, `FINDINGS.md §23`/`§24`). **K1–K4 are pre-registered and NOT yet run**; they need
a maintainer ruling on the grid (§6) before any param is generated. Nothing here changes production.

---

## 1. Why f_κ is being re-opened (and what is *not* being re-opened)

On 2026-07-29 the maintainer supplied the El-Badry+2019 paper page carrying Eq 47. It falsified a claim this
workstream had been leaning on: that `cooling_boost_kappa` pushed evaporation *the wrong way* relative to
El-Badry. It does not. Eq 47's conduction factor is `(C / 6×10⁻⁷ cgs)^{2/7}` — mass-loss **rises** with
conductivity — and `cooling_boost_kappa` multiplies exactly that `C`. The full correction is `FINDINGS.md §23`.

That matters because the "wrong sign" line was used as a *physics* discriminator between the knobs, and it was
wrong. So the honest position is: **f_κ was retired partly on a bad argument, and must be re-read with that
argument deleted.** K0 below does the re-read from committed artifacts.

**What is NOT re-opened.** This is scoped deliberately, because "scrap it all and start over" is the expensive
answer to a cheap problem:

- **The measurements are not contaminated.** `theta5k_summary.csv` (56/56 proper fates, stop_t=5, θ from
  `dictionary.jsonl`) and the per-call `fkappa_leverage.csv` / `kappa_backreaction.csv` snapshots record what
  the code did. The bad argument sat in the *interpretation* layer, not the data layer. Deleting the data would
  destroy good evidence to punish a bad sentence.
- **The E3/E4 contamination grades stand** (`CONTAMINATION.md` ⛔ #1–#4): the blowout-θ₀ calibration, the
  819-cell `stop_t=2` grid, and `cooling_boost_kappa='auto'` remain non-quotable as calibration values. §23
  does not launder them. `'auto'` stays PROVISIONAL.
- **No production change is proposed here.** The default stays `cooling_boost_mode='none'`, f_κ stays 1.0.

---

## 2. K0 — the re-read, DONE (offline, no sims)

```
python docs/dev/transition/pdv-trigger/data/make_kappa_eq47_check.py
```
→ `data/kappa_eq47_check.csv` (three tables). Sources and their register grades are recorded in the CSV header.

### K0.Q1 — does TRINITY's f_κ actually reproduce Eq 47's conduction channel? **YES, at fixed state.**

`data/fkappa_leverage.csv` varies f_κ over a 64× range at two frozen captured states and re-solves the bubble
structure (f_κ=1 is byte-identical there — the harness's own correctness check). The Eq-47 prediction is
`dMdt(f)/dMdt(1) = f^{2/7}` exactly:

| state | fitted exponent | El-Badry 2/7 | max abs error over f ∈ [1,64] | L_cool exponent |
|---|---|---|---|---|
| stiff 5e9/sfe0.01 | **0.2819** | 0.2857 | 1.63% | 0.586 |
| mild cluster | **0.2849** | 0.2857 | 0.34% | 0.669 |

### K0.Q1b — does it hold once the state evolves? **NO — it decays, and the decay is the mechanism.**

Per-call equivalence is necessary but not sufficient (CLAUDE.md rule 5), so the same check runs along the full
f_κ=2 trajectory in `data/kappa_backreaction.csv` (graded *"CLEAN for the f_κ^{2/7} scaling check"*):

| t [Myr] | dMdt ratio | error vs 2^{2/7} | E_b ratio | P_b ratio |
|---|---|---|---|---|
| 1.96e-07 | 1.21752 | **−0.12%** | 1.00000 | 1.00000 |
| 9.00e-04 | 1.14258 | −6.27% | 0.94557 | 0.94888 |
| 2.34e-03 | 1.08122 | **−11.30%** | 0.89455 | 0.90884 |

The boosted arm radiates more, so it **drains E_b**; pressure falls; conduction falls behind the fixed-state
prediction. The C-channel is not failing — the bubble is paying for it. ⚠️ Horizon is t ≲ 2.3e-3 Myr
(CONTAMINATION FLAG-(a)); this is a mechanism statement, not a calibration.

### K0.Q2 — why is there no whole-band f_κ? **Not reach. Condensation fallout.**

`FINDINGS.md §12` recorded "no whole-band f_κ". Re-reading `data/theta5k_fire_map.csv` per dose shows the
recorded *reason* (insufficient leverage/reach) is wrong. **Every band config crosses θ = 0.95 somewhere**
in the grid — peak θ_max ranges 1.04 (`simple_cluster`) to 1.99 (`small_dense_highsfe`). What breaks the band
is CONDENSE/DRAIN fallout at **scattered, non-monotonic** doses:

| f_κ | FIRED | CONDENSE | DRAIN | NOFIRE |
|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 6 |
| 2 | 0 | 0 | 2 | 4 |
| 4 | 4 | 0 | 1 | 1 |
| 6 | 4 | 1 | 1 | 0 |
| 8 | 4 | 1 | 1 | 0 |
| 12 | **5** | 1 | 0 | 0 |
| 16 | 4 | 2 | 0 | 0 |

Denominator is the **6 band configs**, matching `FINDINGS.md §12`'s "5/6": excluded are the two controls
(`fail_repro`, `small_1e6`) *and* `normal_n1e3`, which fires unmodified at f = 1 (θ₀ = 1.047) and so never
tests a knob. The best single dose is f_κ = 12 at 5/6 — reproducing §12 exactly, from the corrected reading.
The individual tracks are non-monotonic —
`pl2_steep` DRAIN@4,6 → FIRED@8,12 → CONDENSE@16; `be_sphere` FIRED@6 → DRAIN@8 → FIRED@12,16;
`simple_cluster` FIRED@4,6 → CONDENSE@8,12,16. **The squeeze:** `pl2_steep` needs f_κ ≥ 8, but
`simple_cluster` condenses from f_κ = 8 up. The coarse grid never samples between them.

Q1b supplies the physical reading of Q2: the same E_b depletion that bends dMdt below `f^{2/7}` is what carries
an arm across the evaporation→condensation boundary (`KAPPA_FREEZE_MECHANISM.md`). **f_κ's Eq-47 fidelity and
its instability are the same effect at different doses.** That is the hypothesis K1–K3 test.

### K0 — the gap this exposes

**f_κ has never been put through the L21b Θ_cum calibration that decided between f_A and f_mix.** The published
decision metric is *band-entry-dose uniformity across density* (`FINDINGS.md §18`), measured on the L21b
Table-1 benches. f_A and f_mix both have a number. f_κ has none — it was cut before that protocol existed. **The
head-to-head is two-way where it should be three-way.** That is K1, and it is the only reason this campaign is
worth >100 HPC arms.

---

## 3. Pre-registered predictions (written BEFORE any arm runs)

Per CLAUDE.md rule 5 / the planning protocol, the gates are defined here, before the grid is generated. If a
number below is wrong, it stays on the page and the finding records the miss — the `SC-0` precedent
(`FINDINGS.md §15k`) is the standard.

**P1 (K1, the headline).** Θ_cum rises as a power law in f_κ with exponent `q` in **0.55–0.70** — the K0.Q1
fixed-state L_cool exponents (0.586 / 0.669), assumed to carry to the integrated metric. Band entry is
Θ_cum = 0.90. With the shared mode-`none` baselines Θ₀ = 0.462 / 0.341 / 0.221 (bench3 / bench2 / bench1,
`data/bench6_analysis.csv`), entry dose `f_κ = (0.90/Θ₀)^{1/q}`:

| q | bench3 | bench2 | bench1 | spread |
|---|---|---|---|---|
| 0.55 | 3.36 | 5.84 | 12.85 | 3.82× |
| **0.60 (central)** | **3.04** | **5.04** | **10.39** | **3.42×** |
| 0.70 | 2.59 | 4.00 | 7.43 | 2.87× |

**So: predicted f_κ band-entry spread ≈ 2.9–3.8×, central 3.4×** — between f_mix's 2.96× (estimated) and f_A's
5.39× (measured). *Falsifiable both ways.*

**P2 (K1).** Because f_κ acts inside the structure ODE, at least one diffuse high-dose arm (bench1 at f_κ ≥ 16)
exits CONDENSE or DRAIN rather than reaching the band — the Q1b back-reaction, at the doses P1 requires. If P1
and P2 are both right, f_κ is *uniform but unreachable*, which is a different failure from f_A's and worth
saying plainly.

**P3 (K2).** No single f_κ fires all 7 fireable theta5 configs, at any spacing — the squeeze in K0.Q2 is a
genuine overlap failure, not a grid-resolution artifact. (Predicting the *null*: K2 is designed to be able to
falsify me.)

**P4 (K3).** The non-monotonic fates are **deterministic** — a bit-identical re-run of a flip arm reproduces its
fate exactly. If P4 fails, the entire theta5k fire map is noise and `FINDINGS.md §12` must be withdrawn.

**P5 (K4).** f_mix band entry measures **within 25%** of the §18 extrapolations (bench2 ≈ 8.16, bench1 ≈ 11.9),
so the 2.96× spread survives as a measurement. The saturation noted in §18 biases extrapolation *low*, so the
measured entry landing *above* the extrapolation is the expected direction of any miss.

---

## 4. The campaign — K1–K4 (105 arms)

Every arm follows the standing protocol: `stop_t = 5`, θ from `dictionary.jsonl` accepted rows, one process per
arm, **prod** (live `cooling_balance` → fire map) + **diag** (`transition_trigger=blowout` → uncensored θ(t) to
blowout = the L21b window) — identical to `runs/make_bench5_params.py` / `make_bench6_params.py`, so the new
numbers drop straight into the §18 table. Single-knob per arm by construction: f_κ arms keep
`cooling_boost_mode=none` and `cooling_boost_fA=1`.

| phase | question | grid | arms |
|---|---|---|---|
| **K1** | the missing third leg of the head-to-head: what is f_κ's L21b band-entry spread? | bench1/2/3 × f_κ ∈ {2,3,4,6,8,12,16,24,32} × {prod, diag} | 54 |
| **K1b** | keep the Phase-5 fire map 3-knob complete at the dense end | bench4/bench5 × f_κ ∈ {2,4,8} × {prod, diag} | 12 |
| **K2** | is the K0.Q2 squeeze real, or just coarse sampling? | the 6 band configs × f_κ ∈ {5,7,9} × prod | 18 |
| **K3** | are the non-monotonic fates physical or nondeterministic? | 5 flip arms × 2 (original + bit-identical repeat) | 10 |
| **K4** | close the f_mix extrapolation the record already owes (§18, maintainer Q4) | bench1/bench2 × f_mix ∈ {12,16} × {prod, diag} | 8 |
| | | **total** | **102** |

Notes on the design choices, so a later visit can argue with them:

- **f_κ = 1 is not re-run.** The bench5 `__none` arms already are it (`cooling_boost_kappa` is gated ×1.0
  exact), and they are the K1 equivalence baseline — see §5.
- **9 doses, not 4.** The deliverable is the *dose–response exponent* `q`, not just the crossing point; a wide
  grid also detects the saturation that made the f_mix extrapolation untrustworthy in the first place.
- **bench4/bench5 are excluded from the decision metric** (K1b is fire-map completeness only) — they fire at low
  dose into a collapse window, so they have no clean L21b breakout window. Same exclusion bench6 applied to f_A.
- **K2 reuses f_κ = 6, 8 from theta5k** rather than re-running them; only 5, 7, 9 are new. It runs the 6 band
  configs only — the two controls and `normal_n1e3` (fires unmodified) cannot change a whole-band verdict.
- **⚡ If a later revision adds `small_1e6` back**, note it is degenerate with `large_diffuse_lowsfe` at early
  time — identical M_cluster, nCore and flat profile give bit-identical early trajectories
  (`CONTAMINATION.md`, `FINDINGS.md §12.6`) — so the pair counts as **one** check, not two.

### The metric, reported two ways (new — closes a §18 registered gap)

`FINDINGS.md §18` found that Θ_cum on the band-setting arms rests substantially on **frozen no-root rows**
(54–67% of Θ_cum on the f_A arms, 7–47% on f_mix). Rather than inherit that silently, **K1 reports both**:

1. `theta_cum_prefire` — all rows, directly comparable to the published f_A / f_mix numbers;
2. `theta_cum_solved` — no-root rows excluded (`data/make_bench_stale_segments.py` already decomposes this).

**Pre-registered:** if the knob *ranking* differs between the two metrics, the ranking is not a result and the
campaign reports "the L21b Θ_cum metric cannot discriminate these knobs" instead of picking a winner.

---

## 5. Gates — pass/fail defined before the runs

| gate | when | bar | fail ⇒ |
|---|---|---|---|
| **G0 baseline** | before generating params | `data/bench6_analysis.csv` Θ₀ = 0.462/0.341/0.221 and §18's band-entry table are reproduced by re-running the two analysis scripts on the committed trajectories | the baseline moved; stop and reconcile before spending HPC time |
| **G1 param emit** | param generation | the bench5/bench6 emit gates pass unchanged (GMC plausibility, `rCloud`(gas) = R_cl within 2%, end-to-end `read_param` load-check) | fix the generator; do not submit |
| **G2 equivalence** | first harvest | the 30 K1/K1b **prod** arms' f_κ = 1 counterparts already exist; no new f_κ=1 arm is run, so the check is that each new arm's t < first-boost-effect trajectory prefix matches its `__none` sibling to ≤ 1e-9 relative | a gated-knob leak; **blocking** — `cooling_boost_kappa` must be ×1.0-exact when unset |
| **G3 compliance** | harvest | ≥ 95% of arms reach `stop_t = 5` or a proper fate (FIRED/CONDENSE/DRAIN/NOFIRE); no freezes | investigate before analysing; a freeze class means `KAPPA_FREEZE_MECHANISM.md` fix #1 regressed |
| **G4 decision** | analysis | K1 band entry measured **in-grid** for all three of bench3/2/1 (no extrapolation — the exact flaw §18 had to flag on f_mix) | extend the grid, or report the spread as *estimated* and say so in the table |
| **G5 honesty** | analysis | both Θ_cum variants (§4) reported; frozen-row share stated per band-setting arm | do not publish a ranking |

**Pre-registered TERMINAL stop** (the `SC-0` pattern): if **G4 passes and f_κ's measured spread is worse than
both** f_A's 5.39× and f_mix's measured spread, **and P3 holds** (K2 finds no whole-band f_κ), then f_κ is
re-closed — this time on correct grounds — and K-phases beyond K4 are **not** to be started. Write the finding,
update `CONTAMINATION.md` and `INDEX.md §1.5`, and stop.

---

## 6. Execution order (nothing is generated until §6.0 is ruled on)

**6.0 — maintainer ruling required, before any param exists.** Three open calls:
   **(a)** the K1 dose grid `{2,3,4,6,8,12,16,24,32}` — accept, or trim/extend;
   **(b)** whether K1b (12 dense arms, fire-map completeness only) is worth the slots;
   **(c)** whether K4's 8 f_mix arms ride along in this campaign (recommended — without them the three-way
   comparison still has an extrapolated leg, and P5 is the cheapest of the five predictions to settle).
   *Deliberately not written yet:* the param generator. Writing it before the grid is ruled on is work thrown
   away if (a) changes.

**6.1** `runs/make_kappa_reopen_params.py` (new, modelled on `make_bench6_params.py`) → `runs/params/bench7/`.
Self-gating per G1. Commit the params.

**6.2** Submit as an sbatch array (`runs/run_bench7.sbatch`, `runs/sync_bench.sh bench7 submit/run/down`);
K2/K3 as a second array against `runs/params/theta5kf/`.

**6.3** Harvest: `runs/harvest_bench5.py "$WS"/outputs/bench7/* --csv runs/data/bench7_summary.csv
--traj-dir runs/data/bench7_traj` (K1/K1b/K4); `runs/harvest_theta_max.py` (K2/K3).

**6.4** Analyse: `data/make_bench7_analysis.py` (new — three-knob band-entry table + both Θ_cum variants),
re-run `data/make_bench_stale_segments.py` over the new trajectories, and re-run
`data/make_kappa_eq47_check.py` with the full-run arms added to Q1b.

**6.5** Write it up: `FINDINGS.md §25`, reconcile `INDEX.md` (§1.5 audit row + doc table), `CONTAMINATION.md`
(register every new artifact with its grade), `PLAN.md` ledger, `REPRODUCE.md` (#48+), regenerate
`MANIFEST.md`, and refresh the three-knob section of `pdvtrigger_report.html`.

---

## 7. Cost and risk

~105 arms at the bench5/bench6 wall-clock profile. The known expensive corner is **dense × high dose**: the
`bench5_fa16_diag` stiffness freeze reproduced on both platforms (`FINDINGS.md §15h`/`§15j`), and f_κ enters the
structure ODE, so its high-dose diffuse arms (bench1 at f_κ ≥ 24) are the analogous risk here. Budget a per-arm
timeout and treat a wall-kill as a recorded non-compliance (G3), not a silent drop.

The scientific risk is **P2**: f_κ may turn out to be the most *uniform* knob whose required doses are
simultaneously *unreachable*. That would be a real result, not a failure — and it is the reason G5 exists, so
the write-up cannot quietly report the uniformity number without the reachability number beside it.

---

## 8. Next-chat handoff (2026-07-29)

**Branch** `feature/pdv-trigger-5` @ `db353aa1`. **Do NOT branch from `origin/main`** — main lags the whole
07-19 → 07-29 close-out. Everything is durable in git; re-derive from the docs + committed CSVs, never from
chat memory.

**State.** K0 is DONE and committed (`FINDINGS §24`, `data/kappa_eq47_check.csv`). K1–K4 are pre-registered
here and **not run**. No production change; default is still `cooling_boost_mode='none'`, f_κ = 1.0.
`pytest` is green except one pre-existing failure in a different workstream
(`test_docs_dev_conventions.py::test_banners[rosette-cf/figs/README.md]`, fails identically at HEAD — leave it).

**The one thing blocking execution: the §6.0 ruling.** Three calls, in priority order:
   (a) the K1 dose grid `{2,3,4,6,8,12,16,24,32}` — accept / trim / extend;
   (b) K1b's 12 dense arms — worth the slots, or drop;
   (c) K4's 8 f_mix arms riding along — **recommended yes** (without them the three-way comparison keeps an
       extrapolated leg, and P5 is the cheapest of the five predictions to settle).
Once ruled: write `runs/make_kappa_reopen_params.py` (model it on `runs/make_bench6_params.py`, self-gating per
G1), then follow §6.1 → §6.5. **Do not write the generator before the ruling** — it is thrown away if (a) moves.

**Standing maintainer questions, still open** (none block K1–K4):
 - **Q1** clause-1 grounds — re-derive from the in-ODE structural asymmetry, or withdraw the framing?
   (`PLAN.md` consistency item 1; the registry info strings wait on it.)
 - **Q2** is `C_f = 1` / `L_leak ≡ 0` expected for the bench configs, or is the leak channel silently
   disabled? (`FINDINGS §18`, `§20`.)
 - **Q3** frozen-row Θ_cum — exclude no-root rows, or carry an uncertainty band? Moves f_A-side numbers too.
   (`PLAN.md` item 6.) **K1 sidesteps this by reporting both variants** (§4), but the published f_A/f_mix
   numbers still need the ruling.
 - **Q5** land the `§16` fallback double-boost fix before the rosette-cf 72-dictionary reduction?
   (`PLAN.md` item 2; `§21` showed 1/36 rosette fm4 fires is bug-dependent.)
 - *(Q4 — the fm{12,16} arms — is no longer separate; it is **K4** here.)*

**Also queued, independent of this campaign:** `PLAN.md` items 8 (rebuild the f_A rationale after `§23`
voided its Eq-47 leg) and 9 (the area-faithful successor — the discriminating experiment, first pass needs
no new sims).
