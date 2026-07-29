# Re-opening `cooling_boost_kappa` — the K0 re-read and the K1–K4 HPC campaign

> ---
>
> ⚠️⚠️ **DEMOTED 2026-07-29 — read this as "could be true, verify before use".** The maintainer no
> longer trusts this workstream's measured numbers without re-measurement, and three corrections in
> five days say that is the right call: `§18` (a metric artifact published "f_mix eliminated" for
> eight days across four documents), `§23` (the "wrong El-Badry sign" argument used to retire f_κ was
> false), `§24` (a correct result with a wrong stated cause). None of these were caught by
> `CONTAMINATION.md` — they were **correct data with a wrong reading**, which a per-artifact grade
> cannot detect.
>
> **The active workstream is now [`docs/dev/transition/kappa-3way/`](../kappa-3way/README.md)** — its
> `report.html` is the source of truth, and its rule is: a number is quotable only if its own
> provenance stamp is dated **on or after 2026-07-29**.
>
> **What this doc is still good for:** the history, the physics reasoning, the design rationale, the
> literature imprints (`LANCASTER_REFERENCE.md`, `ELBADRY_REFERENCE.md` — published values, still
> `[V]`), the measurement rules, and the param/HPC tooling under `runs/` (which stays here and is
> actively used). **What it is not good for:** quoting a measured value. Every Θ_cum, band-entry
> dose, spread, fire map and threshold in here is ⚠️ **VERIFY** until the 294-arm re-run reproduces
> it — see [`../kappa-3way/PROVENANCE.md`](../kappa-3way/PROVENANCE.md).
>
> ---


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

**Status (2026-07-29, updated same day):** 🟡 **submit-ready** — **K0 (the offline re-read) is DONE** and is the
evidence base below (`data/kappa_eq47_check.csv`, `FINDINGS.md §23`/`§24`); the **§6.0 grid ruling has landed**
(§6.0); **gate G0 is CLEARED 11/11** (`data/bench7_gate_g0.csv`); and the **174 params are generated and
committed** (`runs/make_kappa_reopen_params.py` → `runs/params/bench7/`, all four G1 emit gates pass). **The arms
are NOT yet run** — §6.2's `./runs/sync_bench.sh bench7 up|submit` needs cluster access and is the maintainer's
step. Nothing here changes production.

*What changed this visit (2026-07-29, second pass):* the §6.0 ruling was recorded; K4 grew from 8 arms to a
24-arm f_mix **ladder redo** under a flagged reading of the ruling (§6.0(c)); **G6** was added to the gate table
(a tightening the redo makes necessary); §6.1 is DONE; and the P1 prediction table was recomputed at full Θ₀
precision and persisted (§3, `data/bench7_gate_g0.csv`). Campaign total 102 → 118.

*Third pass, same day — the **ALL-FRESH ruling** (§6.2):* the maintainer does not trust the earlier harvests and
requires every number the conclusions rest on to be re-measured today, with a legible timestamp on each
artifact. Two structural consequences: **K2's grid widened** `{5,7,9}` → `{1,…,16}` so `theta5k`'s 2026-07-03
columns stop being an input (campaign **118 → 166 → 174**), and the **`bench5r`/`bench6r` re-run campaigns** were added
so Θ₀ and the f_A/f_mix ladders are today's numbers too (**294 arms** in total). Stamping was extended to the
per-arm trajectory CSVs, the hash files and the analysis outputs, and `data/make_freshness_audit.py` reports
what is fresh. **G0 now doubles as the old-vs-new reproduction gate** — same targets, fresh arms.

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

> 📌 **These numbers are now frozen in a committed CSV, not just on this page.**
> `data/bench7_gate_g0.csv` (table `P1`, built by `data/make_bench7_gate_g0.py`) recomputes the same
> prediction from the **full-precision** measured Θ₀ (0.461806 / 0.340860 / 0.220551) rather than the 3-dp
> values tabulated above, so it reads **3.36 / 5.84 / 12.90 → 3.833×**, **3.04 / 5.04 / 10.42 → 3.427×**,
> **2.59 / 4.00 / 7.455 → 2.874×**. The ≤0.4% offset on the bench1 column is Θ₀ rounding, nothing more —
> **the table above is the pre-registered statement of record** and the CSV is its machine-checkable twin.

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

## 4. The campaign — K1–K4 (**174 arms**, generated and committed 2026-07-29)

Every arm follows the standing protocol: `stop_t = 5`, θ from `dictionary.jsonl` accepted rows, one process per
arm, **prod** (live `cooling_balance` → fire map) + **diag** (`transition_trigger=blowout` → uncensored θ(t) to
blowout = the L21b window) — identical to `runs/make_bench5_params.py` / `make_bench6_params.py`, so the new
numbers drop straight into the §18 table. Single-knob per arm by construction: f_κ arms keep
`cooling_boost_mode=none` and `cooling_boost_fA=1`.

| phase | question | grid | prefix | arms |
|---|---|---|---|---|
| **K1** | the missing third leg of the head-to-head: what is f_κ's L21b band-entry spread? | bench1/2/3 × f_κ ∈ {2,3,4,6,8,12,16,24,32} × {prod, diag} | `k1_` | 54 |
| **K1b** | keep the Phase-5 fire map 3-knob complete at the dense end | bench4/bench5 × f_κ ∈ {2,4,8,12,16} × {prod, diag} | `k1b_` | 20 |
| **K2** | is the K0.Q2 squeeze real, or just coarse sampling? | the 6 band configs × f_κ ∈ {5,7,9} × prod | `k2_` | 18 |
| **K3** | are the non-monotonic fates physical or nondeterministic? | 5 flip arms × 2 (`_a`/`_b`, identical physics) | `k3_` | 10 |
| **K4** | close the f_mix extrapolation the record already owes (§18, maintainer Q4) — **as a full ladder redo**, per the §6.0(c) ruling | bench1/bench2 × f_mix ∈ {2,3,4,8,12,16} × {prod, diag} | `k4_` | 24 |
| | | | **total** | **174** |

**The 5 K3 flip arms**, and the rule that picked them — every cell in `data/theta5k_fire_map.csv` whose fate
reverses against its dose neighbours. Two are isolated single-cell reversals; three are the grid-edge/onset
reversals the K0.Q2 squeeze actually rests on:

| arm | the reversal | kind |
|---|---|---|
| `be_sphere` @ f_κ=8 | FIRED@6 → **DRAIN@8** → FIRED@12 | isolated |
| `small_dense_highsfe` @ f_κ=6 | FIRED@4 → **CONDENSE@6** → FIRED@8 | isolated |
| `pl2_steep` @ f_κ=16 | FIRED@12 → **CONDENSE@16** | grid edge |
| `normal_n1e3` @ f_κ=16 | FIRED@12 → **DRAIN@16** | grid edge |
| `simple_cluster` @ f_κ=8 | FIRED@6 → **CONDENSE@8** | the squeeze's upper limit |

Notes on the design choices, so a later visit can argue with them:

- **f_κ = 1 is not re-run.** The bench5 `__none` arms already are it (`cooling_boost_kappa` is gated ×1.0
  exact), and they are the K1 equivalence baseline — see §5.
- **9 doses, not 4.** The deliverable is the *dose–response exponent* `q`, not just the crossing point; a wide
  grid also detects the saturation that made the f_mix extrapolation untrustworthy in the first place.
- **bench4/bench5 are excluded from the decision metric** (K1b is fire-map completeness only) — they fire at low
  dose into a collapse window, so they have no clean L21b breakout window. Same exclusion bench6 applied to f_A.
- **K2 re-measures theta5k's whole f_κ grid and fills it in** (widened from the original `{5,7,9}` by the
  2026-07-29 ALL-FRESH ruling — see §6.2). It runs the 6 band configs only: the two controls and
  `normal_n1e3` (fires unmodified) cannot change a whole-band verdict. The `f_κ = 1` column is a gated
  ×1.0 exact no-op, i.e. the config's native Θ₀ — included so the K2 fire map is self-contained.
- **K4 is a ladder redo, not a ride-along** (§6.0(c) ruling, 2026-07-29). Re-running f_mix ∈ {2,3,4,8} alongside
  the new {12,16} costs 16 extra arms and buys two things the 8-arm version could not: f_mix band entry becomes
  **measured inside one campaign, one code state, one reduce** — the same in-grid bar G4 sets for f_κ, and the
  fix for exactly the flaw `§18` had to flag — and the overlap with the 2026-07-19 bench6 ladder becomes a
  cross-campaign reproduction check (**G6**). `normal_n1e3` and the dense benches are not in K4: `§18`'s
  extrapolated legs are bench2 and bench1 only. ⚠️ The ruling's wording ("no, redo if possible") was read, not
  confirmed — see the flag in §6.0(c).
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
| **G0 baseline** ✅ **CLEARED 2026-07-29, 11/11** | before generating params | `data/bench6_analysis.csv` Θ₀ = 0.462/0.341/0.221 and §18's band-entry table are reproduced by re-running the two analysis scripts on the committed trajectories | the baseline moved; stop and reconcile before spending HPC time |
| **G1 param emit** ✅ **CLEARED 2026-07-29, 4/4** | param generation | the bench5/bench6 emit gates pass unchanged (GMC plausibility, `rCloud`(gas) = R_cl within 2%, end-to-end `read_param` load-check) | fix the generator; do not submit |
| **G2 equivalence** | first harvest | the 30 K1/K1b **prod** arms' f_κ = 1 counterparts already exist; no new f_κ=1 arm is run, so the check is that each new arm's t < first-boost-effect trajectory prefix matches its `__none` sibling to ≤ 1e-9 relative | a gated-knob leak; **blocking** — `cooling_boost_kappa` must be ×1.0-exact when unset |
| **G3 compliance** | harvest | ≥ 95% of arms reach `stop_t = 5` or a proper fate (FIRED/CONDENSE/DRAIN/NOFIRE); no freezes | investigate before analysing; a freeze class means `KAPPA_FREEZE_MECHANISM.md` fix #1 regressed |
| **G4 decision** | analysis | K1 band entry measured **in-grid** for all three of bench3/2/1 (no extrapolation — the exact flaw §18 had to flag on f_mix) | extend the grid, or report the spread as *estimated* and say so in the table |
| **G5 honesty** | analysis | both Θ_cum variants (§4) reported; frozen-row share stated per band-setting arm | do not publish a ranking |
| **G6 fm reproduction** *(added 2026-07-29 with the §6.0(c) redo — a tightening, not a loosening; G0–G5 are unchanged)* | analysis | K4's overlapping f_mix doses {2,3,4,8} reproduce the bench6 ladder (bench1 0.380/0.494/0.579/0.767, bench2 0.533/0.649/0.727/0.895) to ≤2% on Θ_cum, and no fire label flips | the bench6 and bench7 f_mix arms are **not** one measurement; report K4's ladder standalone and say so — do **not** merge the two campaigns' points into one band-entry fit |

**Why G6 exists.** The 8-arm ride-along would have stitched fm{12,16} onto a 2026-07-19 harvest; the redo
instead re-measures the whole ladder, which *removes* the stitching risk but *introduces* a new one — that the
two campaigns silently disagree. G6 turns that into a checkable statement. If the ruling is flipped back to the
ride-along (`F_MIX_K4 = ["12","16"]`), G6 becomes inapplicable and should be struck.

**Pre-registered TERMINAL stop** (the `SC-0` pattern): if **G4 passes and f_κ's measured spread is worse than
both** f_A's 5.39× and f_mix's measured spread, **and P3 holds** (K2 finds no whole-band f_κ), then f_κ is
re-closed — this time on correct grounds — and K-phases beyond K4 are **not** to be started. Write the finding,
update `CONTAMINATION.md` and `INDEX.md §1.5`, and stop.

---

## 6. Execution order (nothing is generated until §6.0 is ruled on)

**6.0 — the maintainer ruling. ✅ LANDED 2026-07-29.**

| call | ruling | what was done |
|---|---|---|
| **(a)** the K1 dose grid `{2,3,4,6,8,12,16,24,32}` | **accepted as written** | `F_KAPPA_K1` in the generator, verbatim; 54 arms |
| **(b)** K1b's 12 dense arms | **keep** | `K1B_BENCHES` × `{2,4,8}` × prod/diag; 12 arms |
| **(c)** K4's 8 f_mix arms riding along | **"no, redo if possible"** | read as: *no* to the ride-along, *yes* to a full ladder redo — bench1/bench2 × f_mix {2,3,4,8,12,16} × prod/diag, 24 arms, plus **G6** |

> ⚠️ **(c) is a READING, not a confirmation — check it before submitting.** "no, redo if possible" is
> two-way: it can mean *drop K4 and redo f_mix later as its own campaign*, or *don't just ride along —
> redo the ladder properly here*. The second was taken, because a redo is possible at param cost only and
> it is what makes f_mix's band entry **measured** rather than extrapolated (`§18`'s standing flaw). The
> choice is one line in `runs/make_kappa_reopen_params.py`:
> `F_MIX_K4 = ["2","3","4","8","12","16"]` (174 arms, current) · `[]` (142 arms, K4 dropped, P5 recorded
> **NOT RUN** — not *missed*) · `["12","16"]` (158 arms, the literal ride-along). Nothing else changes;
> `submit` re-sizes the array from the param count on its own.

**6.1 ✅ DONE 2026-07-29.** `runs/make_kappa_reopen_params.py` (new, modelled on `make_bench6_params.py`) →
**174 committed params in `runs/params/bench7/`**. Self-gating per G1 — all four emit gates pass (GMC
plausibility on every arm incl. the theta5 configs, the exact L21b mapping ≤2%, an end-to-end `read_param`
load-check on all 174 files, and a count/uniqueness assertion). **All five K-phases emit into that ONE
directory** — a phase is just a filename prefix (`k1_…`, `k1b_…`, `k2_…`, `k3_…`, `k4_…`), so the campaign is
one array, one reduce, one download rather than two of everything.

`test/test_bench7_params.py` (182 cases) pins the set against its builder: byte-identical regeneration, the
per-phase counts, `stop_t = 5` / `model_name` / `path2output` on every arm, **single-knob** by construction,
the prod-vs-diag `transition_trigger` split, the K3 pairs differing in nothing but their names (P4 rests on
that), and every bench arm sitting on the same cloud as its bench5 `__none` sibling (G2 rests on that).

---

### 6.2 — the ALL-FRESH run order (maintainer ruling, 2026-07-29)

> **The ruling.** *"I do not really trust the previous runs and I would like very fresh ones… everything I
> want will be new numerically… not the csv or files or conclusion that are from before."* So the campaign is
> no longer bench7 alone. **Every number the bench7 conclusions rest on is re-measured**, and each artifact
> carries the UTC moment it was produced.

**What that changed in the design** (both already committed):

1. **K2's dose grid widened `{5,7,9}` → `{1,2,3,4,5,6,7,8,9,12,16}`** (18 → 66 arms; campaign 118 → **166 → 174**).
   The old grid measured only the three *new* doses and reused `theta5k`'s 2026-07-03 columns for the rest —
   so the P3 whole-band verdict would have been part today's data and part four weeks old. K2 now re-measures
   the entire f_κ fire map for the 6 band configs, `f_κ = 1` baseline column included. **`theta5k` is no
   longer an input to any bench7 conclusion.**
2. **Two new campaigns, `bench5r` and `bench6r`** — the *same committed params* as bench5/bench6, re-run
   today, landing under **fresh names** (`bench5r_summary.csv` + `bench5r_traj/`, etc.). Nothing older is
   overwritten, so old-vs-new is a file diff. They also collect bench7's four extra trajectory columns, which
   the 07-19 harvests never captured. Without these, Θ₀ and the f_A/f_mix ladders — i.e. two of the three legs
   of the head-to-head — would still be 07-19 numbers.

**Order matters: the baselines first.** `bench5r` supplies Θ₀ (the f_κ=1 / f_mix=1 column) and re-clears G0
against today's arms. If G0 **fails** on fresh data, that is a finding about the 07-19 result and it should be
reconciled *before* bench7's 174 arms are read — though bench7 can run concurrently, since nothing in its
submission depends on the outcome.

```bash
# ── 0. get the code onto the cluster (once) ────────────────────────────────────
git pull                                        # laptop: pick up feature/pdv-trigger-5b
./runs/sync_bench.sh bench7 up                  # `up` is campaign-agnostic — one pull serves all

# ── 1. the baselines, re-run today (60 + 60 arms) ─────────────────────────────
./runs/sync_bench.sh bench5r submit             # auto-sized --array=1-60 over params/bench5/
./runs/sync_bench.sh bench6r submit             # auto-sized --array=1-60 over params/bench6/
./runs/sync_bench.sh bench5r watch              # (Ctrl-C to stop watching; the array keeps running)

# ── 2. the campaign itself (174 arms) — submit any time after step 0 ──────────
./runs/sync_bench.sh bench7 submit              # auto-sized --array=1-174 over params/bench7/
./runs/sync_bench.sh bench7 watch

# ── 3. reduce + download, once each array is DONE (⚠️ THE REDUCE IS ONE-SHOT) ──
./runs/sync_bench.sh bench5r reduce  && ./runs/sync_bench.sh bench5r down
./runs/sync_bench.sh bench6r reduce  && ./runs/sync_bench.sh bench6r down
./runs/sync_bench.sh bench7  reduce  && ./runs/sync_bench.sh bench7  down

# ── 4. re-derive everything from the fresh data, then prove it IS fresh ───────
python data/make_bench5_analysis.py             # auto-prefers bench5r_*; prints SOURCES READ
python data/make_bench6_analysis.py             # auto-prefers bench5r_*/bench6r_*
python data/make_bench7_gate_g0.py              # G0 re-run vs the SAME targets, on today's arms
python data/make_freshness_audit.py             # the receipt: what is FRESH / OLD / UNSTAMPED
git add -A && git commit && git push            # commit the fresh CSVs + trajectories
```

**Every artifact is timestamped.** `_stamp.py` writes `# generated <UTC ISO8601> | builder <x> | code <sha>`
as the first line of each output, and as of 2026-07-29 that stamp is on the **per-arm trajectory CSVs** too
(`harvest_bench5.py`), on `<campaign>_hashes.csv` (`sync_bench.sh`), and on the three analysis outputs — it
was previously only on the summary. `data/make_freshness_audit.py` reads every stamp back and prints the
FRESH / OLD / UNSTAMPED roll-up, so "is this today's number?" is one command, not an inspection. It also
flags `+dirty` artifacts: fresh, but built from an uncommitted tree, so **regenerate from a clean tree before
quoting**. The K3 determinism hash is taken over **non-comment lines**, so stamping cannot make two identical
runs look different.

**Cost.** 294 arms total (174 + 60 + 60) at `--time=1:30:00` each. The 07-19 bench5 evidence
(`data/bench5_durations.csv`) puts the longest compliant arm at 64 min under 3-worker contention, so the
array is the wall-clock constraint, not any single job.

**6.3 — the campaign command reference** (unchanged; `submit` auto-sizes `--array` from the committed param
count, so a grid change needs no edit):

```
./runs/sync_bench.sh <campaign> up        # git pull the committed code on the cluster
./runs/sync_bench.sh <campaign> submit    # auto-sized array
./runs/sync_bench.sh <campaign> watch     # queue + newest task log
./runs/sync_bench.sh <campaign> reduce    # multi-GB jsonl -> small CSVs, ON HPC
./runs/sync_bench.sh <campaign> down      # ships ONLY the reduced CSVs into runs/data/
```

`reduce` runs `harvest_bench5.py` on the cluster and writes three things: `bench7_summary.csv` (the fire map),
`bench7_traj/` (per-arm θ(t), ≤4000 rows, log-t downsampled, endpoints kept), and `bench7_hashes.csv` (sha256
of each reduced trajectory — this is what makes **K3's determinism claim (P4) checkable** without shipping a
raw dictionary down: two runs of one param must hash identically). The raw `dictionary.jsonl` files never
leave gpfs.

> ⚠️ **The reduce is one-shot — declare the columns BEFORE the first one.** gpfs workspaces get cleaned and
> the raw arms do not come back; this already cost the workstream once (theta5s's arms were lost to a `/tmp`
> wipe and `dMdt` had to be salvaged in a scramble — `harvest_bench5.py`'s docstring). The six default
> trajectory columns do **not** carry `Pb` or `bubble_dMdt`, and **P2 and the K0.Q1b back-reaction both read
> them**, so `sync_bench.sh` passes
> `--extra-cols Pb,bubble_dMdt,bubble_L2Conduction,bubble_L3Intermediate` for bench7. Do not hand-run the
> harvest without those columns. If a later K-phase needs another field, add it to that list *before*
> submitting, not after.

**6.4** Analyse: `data/make_bench7_analysis.py` (new — three-knob band-entry table + both Θ_cum variants),
re-run `data/make_bench_stale_segments.py` over the new trajectories, and re-run
`data/make_kappa_eq47_check.py` with the full-run arms added to Q1b. K3's determinism check is a diff of the
paired rows in `bench7_hashes.csv` — no new harness needed.

**6.5** Write it up: `FINDINGS.md §25`, reconcile `INDEX.md` (§1.5 audit row + doc table), `CONTAMINATION.md`
(register every new artifact with its grade), `PLAN.md` ledger, `REPRODUCE.md` (#48+), regenerate
`MANIFEST.md`, and refresh the three-knob section of `pdvtrigger_report.html`.

---

## 7. Cost and risk

174 arms, plus the 120 baseline arms of `bench5r`/`bench6r` (§6.2) = **294**, at the bench5/bench6 wall-clock profile. The known expensive corner is **dense × high dose**: the
`bench5_fa16_diag` stiffness freeze reproduced on both platforms (`FINDINGS.md §15h`/`§15j`), and f_κ enters the
structure ODE, so its high-dose diffuse arms (bench1 at f_κ ≥ 24) are the analogous risk here. Budget a per-arm
timeout and treat a wall-kill as a recorded non-compliance (G3), not a silent drop.

The scientific risk is **P2**: f_κ may turn out to be the most *uniform* knob whose required doses are
simultaneously *unreachable*. That would be a real result, not a failure — and it is the reason G5 exists, so
the write-up cannot quietly report the uniformity number without the reachability number beside it.

---

## 8. Next-chat handoff (2026-07-29, rewritten after the §6.0 ruling and the ALL-FRESH ruling)

**Base: `origin/main`.** `feature/pdv-trigger-5` was merged to main on 2026-07-29 (PR #731, merge `3264d79e`);
the bench7 HPC tooling followed via PR #732 (merge `c5b2c01`, commit `7f1ca95`). Main carries both, and is the
correct branch point — the earlier "do NOT branch from main" warning is **retired**. Everything is durable in
git; re-derive from the docs + committed CSVs, never from chat memory.

**State.** K0 DONE (`FINDINGS §24`, `data/kappa_eq47_check.csv` — re-verified 2026-07-29: it regenerates
**byte-identically**). HPC tooling DONE (`runs/run_bench7.sbatch`, `bench7` in `runs/sync_bench.sh` — §6.2).
**§6.0 ruled** (§6.0 table). **G0 CLEARED 11/11** (`data/bench7_gate_g0.csv`, rebuild with
`python data/make_bench7_gate_g0.py`; exits non-zero if any row fails). **G1 CLEARED 4/4 and the 174 params are
committed** (`runs/params/bench7/`). No production change; default is still `cooling_boost_mode='none'`,
f_κ = 1.0. Full `pytest` is **896 passed / 0 failed** — including
`test_docs_dev_conventions.py::test_banners[rosette-cf/figs/README.md]`, which the previous handoff recorded as
a standing failure: **it passes now**, so that caveat is retired.

**The one thing blocking execution is cluster access, not a decision.** §6.2's loop must run from a machine
that can `ssh helix`; the container this was prepared in has no `ssh` at all. **§6.2 now holds the full
ALL-FRESH run order — follow it there, not here** (294 arms: `bench5r` 60 + `bench6r` 60 + `bench7` 174, then
the four re-derive commands and `make_freshness_audit.py`).

**Two things to settle BEFORE `submit`, both cheap:**
 1. **§6.0(c) is a reading, not a confirmation** — K4 is currently the 24-arm ladder redo. If that is wrong,
    change `F_MIX_K4` in `runs/make_kappa_reopen_params.py` (`[]` → 150 arms · `["12","16"]` → 158 arms),
    re-run the builder, re-run `pytest test/test_bench7_params.py` (its `PHASE_COUNTS` must be updated to
    match), and re-commit. After `submit` this is no longer free.
 2. **The reduce is one-shot.** `sync_bench.sh` already passes
    `--extra-cols Pb,bubble_dMdt,bubble_L2Conduction,bubble_L3Intermediate`. If any analysis you intend to run
    needs another `dictionary.jsonl` field, add it to that list *now*.

Then §6.4 → §6.5: write `data/make_bench7_analysis.py` (three-knob band-entry table + **both** Θ_cum variants
per §4/G5), re-run `data/make_bench_stale_segments.py` over the new trajectories, extend
`data/make_kappa_eq47_check.py`'s Q1b with the full-run arms, check P4 by diffing the paired `k3_*_a`/`k3_*_b`
rows of `bench7_hashes.csv`, and check G6 against the fm ladder. ⚠️ Under the ALL-FRESH ruling every one
of those steps must read the **fresh** artifacts: `make_bench5_analysis.py` / `make_bench6_analysis.py` /
`make_bench7_gate_g0.py` auto-prefer `bench5r_*`/`bench6r_*` and print the SOURCES line they used —
check it. G6 then compares K4's ladder against **bench6r**, not the 07-19 bench6. Record any prediction miss as a miss
(the SC-0 pattern, `FINDINGS §15k`); if G4 cannot be met in-grid, write **"estimated"** in the table rather
than extrapolating silently.

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
