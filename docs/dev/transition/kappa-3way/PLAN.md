# PLAN — what happened, what to do, what we will do

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

**Status (2026-07-30):** 🟡 campaign COMPLETE — 294/294 arms ran and are reduced. Results in
`FINDINGS.md`; §5's analysis plan is executed. G0 **failed 2/11** (wall-clock truncation, §1 there).
Remaining work is §9 of `FINDINGS.md`, chiefly re-running the 21 truncated arms with more walltime.

---

## 1. WHAT HAPPENED

### 1.1 The problem this program has been chasing since 2026-06-24

TRINITY switches a feedback bubble from energy-driven to momentum-driven when interface cooling drains the
mechanical luminosity — the `cooling_balance` trigger, θ = L_cool/L_mech ≥ 0.95. TRINITY's native 1-D θ
**undershoots** the observed value for most clouds, so realistic GMCs never fire. Lancaster 2021b measures
Θ ≈ 0.90–0.99 in 3-D. That gap is the problem.

The accepted direction (corrected 2026-07-01, after a wrong turn): **θ is an OUTPUT, not an input.** Boost
the cooling *mechanism* and let the solved bubble produce θ; use El-Badry's closed form and Lancaster's band
as the *calibration target*; prefer a **single physical constant** over a fitted f(n).

### 1.2 The three candidate knobs

| knob | what it multiplies | acts where |
|---|---|---|
| `cooling_boost_fA` (f_A) | the L2+L3 interface source terms | **in-solve** — T(r) and Ṁ respond |
| `cooling_boost_mode='multiplier'` (f_mix) | L_cool, as a scalar | **frozen** structure by construction |
| `cooling_boost_kappa` (f_κ) | the Spitzer conduction coefficient C | **in-structure** — θ emerges as an output |

### 1.3 How the choice is supposed to be made

The **L21b Θ_cum band-entry calibration**. For each knob and each of three clean-blowout benches, find the
dose at which Θ_cum first enters [0.90, 0.99]; then compute **the spread of that dose across density**
(max/min). Small spread ⇒ the knob behaves like one physical constant. Large spread ⇒ it is a fitted
function of density wearing a constant's clothes.

### 1.4 What went wrong, three times, and what it teaches

This is the part that justifies the whole all-fresh approach, so it is stated plainly.

**(a) The metric artifact (`§17` → `§18`, 2026-07-27/28).** bench6's Θ_cum numerator integrated the raw
`Lcool` trajectory column. Under `cooling_boost_mode='multiplier'` that column **does not carry the boost**.
So every f_mix arm was scored as if unboosted. The published conclusion — *"f_mix ELIMINATED as a
calibration knob: never reaches the band, wrong-sign dose-response"* — was an artifact of the measurement.
Corrected (integrate θ·L_mech, the effective loss), f_mix's dose–response is **monotone** (bench1 Θ_cum
0.221 → 0.767 over fm 1→8) and the head-to-head **inverts**: f_mix spread 2.96× vs f_A's 5.39×.
*Lesson: a conclusion survived eight days and four documents because nobody re-derived it.*

**(b) The falsified physics claim (`§23`, 2026-07-29).** The argument used to retire f_κ — *"it pushes
evaporation the wrong way relative to El-Badry"* — is **false**. Eq 47 carries `(C/6×10⁻⁷ cgs)^{2/7}`:
mass-loss **rises** with conductivity, and f_κ multiplies exactly that C. TRINITY reproduces the scaling to
0.34–1.63% over f_κ ∈ [1,64]. Five sites were corrected, including a shipped `registry.py` info string.
*Lesson: a knob was retired on an argument nobody had checked against the paper.*

**(c) The mis-attributed cause (`§24`, 2026-07-29).** `§12` recorded "no whole-band f_κ" and blamed
insufficient reach. Re-reading the fire map: **every band config crosses θ = 0.95 somewhere**. The band
breaks on scattered, non-monotonic CONDENSE/DRAIN fallout — the condensation boundary — not on reach. The
5/6-at-f_κ=12 *result* was reproduced exactly; only the *why* was wrong.
*Lesson: correct data, wrong reading. The register of contamination grades cannot catch this class.*

### 1.5 The gap that opened, and this workstream's reason to exist

Put (b) and (c) together and the position is: **f_κ was retired partly on a bad argument, and its recorded
failure mode was mis-diagnosed.** Meanwhile f_A and f_mix both have L21b band-entry numbers and f_κ has
**none** — it has never been through the calibration that decided between the other two.

> **The published head-to-head is two-way where it should be three-way.**

### 1.6 What was already done, before this directory existed

- **K0 — the offline re-read** (`§24`, `pdv-trigger/data/kappa_eq47_check.csv`): Eq-47 match at fixed state,
  its decay under back-reaction, and the squeeze re-attribution. Done; regenerates byte-identically.
- **Campaign design + pre-registration** (`pdv-trigger/KAPPA_REOPEN_PLAN.md`): predictions P1–P5, gates
  G0–G6, a pre-registered terminal stop.
- **Gate G0 cleared 11/11** — the published Θ₀ and band-entry table recompute exactly from the trajectories
  they were derived from (`pdv-trigger/data/bench7_gate_g0.csv`).
- **Gate G1 cleared 4/4; 174 params emitted and committed**, pinned by `test/test_bench7_params.py`.
- **Freshness plumbing**: stamps extended to trajectory CSVs, hash files and analysis outputs;
  `make_freshness_audit.py`; source-precedence in every analysis builder.

### 1.7 Why the old numbers are no longer trusted — and what that changed

The maintainer's ruling, 2026-07-29: *"I do not really trust the previous runs… everything I want will be
new numerically… not the csv or files or conclusion that are from before."*

Given §1.4, that is the right call: the failure mode here is not corrupt data, it is layered readings, and
the only reliable fix is one campaign, one code state, one reduce. Three design consequences:

| # | change | before → after |
|---|---|---|
| 1 | **K2's grid widened** so `theta5k` (2026-07-03) stops being an input to the whole-band verdict | f_κ {5,7,9} → **{1,2,3,4,5,6,7,8,9,12,16}**, 18 → 66 arms |
| 2 | **The L21b baselines re-run** as `bench5r`/`bench6r`, so Θ₀ and the f_A/f_mix ladders are today's numbers | 0 → **120 arms** |
| 3 | **K1b extended to 12 and 16** — it was the only grid stopping short, leaving the dense end dark exactly where K1 and K2 are densest | f_κ {2,4,8} → **{2,4,8,12,16}**, 12 → 20 arms |

**Campaign: 174 (bench7) + 60 (bench5r) + 60 (bench6r) = 294 arms.**

---

## 2. WHAT TO DO — the campaign

All arms: `stop_t = 5 Myr`, one process each, **single-knob by construction** (the f_κ arms hold
`cooling_boost_mode=none` and f_A=1; the f_mix arms hold f_κ=1 and f_A=1), and the two-arm protocol —
**production** (live `cooling_balance` → the fire map) plus **diagnostic** (`transition_trigger blowout` →
uncensored θ(t) across the L21b window).

### 2.1 `bench7` — 174 arms, one params dir, one array, one reduce

A K-phase is only a filename prefix, so the whole campaign submits and reduces once.

| phase | prefix | grid | arms | the question |
|---|---|---|---|---|
| **K1** | `k1_` | bench1/2/3 × f_κ {2,3,4,6,8,12,16,24,32} × {prod,diag} | 54 | **the missing third leg** — f_κ's band-entry spread |
| **K1b** | `k1b_` | bench4/bench5 × f_κ {2,4,8,12,16} × {prod,diag} | 20 | dense-end fire map, now comparable at 12/16 |
| **K2** | `k2_` | 6 band configs × f_κ {1,2,3,4,5,6,7,8,9,12,16} × prod | 66 | the whole f_κ fire map + the condensation squeeze |
| **K3** | `k3_` | 5 fate-flip arms × 2 (`_a`/`_b`) × prod | 10 | are the non-monotonic fates physical or nondeterministic? |
| **K4** | `k4_` | bench1/bench2 × f_mix {2,3,4,8,12,16} × {prod,diag} | 24 | the f_mix ladder, measured in-grid instead of extrapolated |

**K3's five arms** and the rule that picked them — every cell in the fire map whose fate reverses against
its dose neighbours:

| arm | reversal | kind |
|---|---|---|
| `be_sphere` @ f_κ=8 | FIRED@6 → **DRAIN@8** → FIRED@12 | isolated |
| `small_dense_highsfe` @ f_κ=6 | FIRED@4 → **CONDENSE@6** → FIRED@8 | isolated |
| `pl2_steep` @ f_κ=16 | FIRED@12 → **CONDENSE@16** | grid edge |
| `normal_n1e3` @ f_κ=16 | FIRED@12 → **DRAIN@16** | grid edge |
| `simple_cluster` @ f_κ=8 | FIRED@6 → **CONDENSE@8** | the squeeze's upper limit |

Each is emitted **twice**, differing in nothing but `model_name` and `path2output` (both `input_admin`,
neither physics). P4 is then a diff of the paired rows in `bench7_hashes.csv`.

### 2.2 `bench5r` + `bench6r` — 120 arms, the baselines re-measured

The *same committed params* as bench5/bench6, re-run today, landing under **fresh names**
(`bench5r_summary.csv` + `bench5r_traj/`, likewise `bench6r_*`). Nothing older is overwritten, so old-vs-new
is a file diff. They also capture bench7's four extra trajectory columns, which the 07-19 harvests never did.

Without these, two of the three legs of the head-to-head would still be 2026-07-19 numbers.

### 2.3 ⚠️ One open call, cheap to change, expensive after `submit`

**K4's 24-arm ladder rests on a *reading*, not a confirmation.** The ruling on whether the f_mix arms ride
along was *"no, redo if possible"* — which can mean *drop K4, redo f_mix separately later*, or *don't just
ride along, redo the ladder properly here*. The second was taken, because a redo costs params only and it is
what makes f_mix's band entry **measured** rather than extrapolated — the standing flaw from `§18`.

One line in `pdv-trigger/runs/make_kappa_reopen_params.py`:

```python
F_MIX_K4 = ["2", "3", "4", "8", "12", "16"]   # current — 174 arms
# F_MIX_K4 = []                                # drop K4 — 150 arms, P5 recorded NOT RUN (not missed)
# F_MIX_K4 = ["12", "16"]                      # the literal ride-along — 158 arms
```

Change it → re-run the builder → update `PHASE_COUNTS` in `test/test_bench7_params.py` → re-commit.
After `submit`, this is no longer free.

### 2.4 ⚠️ The reduce is ONE-SHOT

gpfs workspaces are cleaned; the raw `dictionary.jsonl` arms do not come back. `sync_bench.sh` already
declares `--extra-cols Pb,bubble_dMdt,bubble_L2Conduction,bubble_L3Intermediate` — needed by P2 and by the
K0.Q1b back-reaction, and not carried by the six default trajectory columns. **If any analysis you intend to
run needs another field, add it now.**

---

## 3. THE PRE-REGISTERED PREDICTIONS

Frozen before any arm runs, in `pdv-trigger/data/bench7_gate_g0.csv` with `verdict=PENDING`. A miss is
**recorded as a miss**, never re-negotiated (the SC-0 pattern, `§15k`).

| # | prediction | how it is decided |
|---|---|---|
| **P1** | f_κ band entry follows `(0.90/Θ₀)^{1/q}` with q ∈ [0.55, 0.70] ⇒ **spread ≈ 2.9–3.8×, central 3.4×** — between f_mix's 2.96× and f_A's 5.39× | K1's measured entry doses |
| **P2** | Ṁ **rises** with f_κ (Eq-47 C-channel) but the ratio **decays** along a full run as E_b drains | the `bubble_dMdt` + `Pb` trajectory columns |
| **P3** | the K0.Q2 squeeze is real: no single f_κ fires all 6 band configs, and the failures are CONDENSE/DRAIN, not NOFIRE | K2's 66-arm fire map |
| **P4** | the non-monotonic fates are **deterministic** — identical params give identical trajectories | K3's paired hashes |
| **P5** | f_mix's band entry is reached **in-grid** by fm ≤ 16 on bench1 and bench2 | K4's ladder |

P1's predicted doses per bench (from the VERIFY Θ₀; will be recomputed from the fresh Θ₀ and **both**
recorded):

| q | bench3 | bench2 | bench1 | spread |
|---|---|---|---|---|
| 0.55 | 3.36 | 5.84 | 12.85 | 3.82× |
| **0.60 (central)** | **3.04** | **5.04** | **10.39** | **3.42×** |
| 0.70 | 2.59 | 4.00 | 7.43 | 2.87× |

---

## 4. THE RUN ORDER — exact commands

Run from the repo root on a machine that can `ssh helix`. Baselines first: `bench5r` supplies Θ₀ and
re-clears G0 against today's arms. bench7 can go in parallel — nothing in its submission depends on the
outcome.

```bash
# ── 0. get the code onto the cluster (once) ────────────────────────────────────
git pull                                        # branch feature/pdv-trigger-5b
./docs/dev/transition/pdv-trigger/runs/sync_bench.sh bench7 up   # `up` serves all campaigns

# ── 1. submit: baselines (60 + 60) and the campaign (174) ─────────────────────
cd docs/dev/transition/pdv-trigger/runs
./sync_bench.sh bench5r submit                  # auto-sized --array=1-60  over params/bench5/
./sync_bench.sh bench6r submit                  # auto-sized --array=1-60  over params/bench6/
./sync_bench.sh bench7  submit                  # auto-sized --array=1-174 over params/bench7/
./sync_bench.sh bench7  watch                   # Ctrl-C stops watching, not the array

# ── 2. reduce + download once each array is DONE (⚠️ ONE-SHOT) ────────────────
./sync_bench.sh bench5r reduce && ./sync_bench.sh bench5r down
./sync_bench.sh bench6r reduce && ./sync_bench.sh bench6r down
./sync_bench.sh bench7  reduce && ./sync_bench.sh bench7  down

# ── 3. re-derive everything from the fresh data, then prove it IS fresh ───────
cd ../../../../..                               # back to repo root
python docs/dev/transition/pdv-trigger/data/make_bench5_analysis.py    # auto-prefers bench5r_*
python docs/dev/transition/pdv-trigger/data/make_bench6_analysis.py    # auto-prefers bench5r_*/bench6r_*
python docs/dev/transition/pdv-trigger/data/make_bench7_gate_g0.py     # G0 vs the SAME targets, fresh arms
python docs/dev/transition/pdv-trigger/data/make_freshness_audit.py    # the receipt
python docs/dev/transition/kappa-3way/make_report.py                   # rebuild the source of truth
git add -A && git commit && git push
```

Every analysis builder prints a `SOURCES READ:` line and writes it into its CSV header. **Check it** — that
line is how you know whether you are looking at fresh or 07-19 data.

**Cost.** 294 arms at `--time=1:30:00`. The longest compliant bench5 arm was 64 min under 3-worker
contention (`pdv-trigger/data/bench5_durations.csv`, VERIFY), so the array is the wall-clock constraint, not
any single job. The known expensive corner is **diffuse × high dose** (bench1 at f_κ ≥ 24): f_κ enters the
structure ODE, making those the analogue of the `bench5_fa16_diag` stiffness freeze. A wall-kill is a
**recorded G3 non-compliance**, not a silent drop; resubmit those ids with `--time=3:00:00`.

---

## 5. WHAT WE WILL DO — the analysis

### 5.1 The gates, in order

| gate | bar | fail ⇒ |
|---|---|---|
| **G0** ✅ cleared 11/11 pre-run; **re-runs against fresh arms** | Θ₀ and the `§18` band-entry table reproduce at the pre-registered tolerances | the 07-19 result did not reproduce — record both numbers and the diff **before** reading anything downstream |
| **G1** ✅ cleared 4/4 | GMC plausibility on every arm, exact L21b mapping ≤2%, end-to-end `read_param` load-check, count/uniqueness | fix the builder; do not submit |
| **G2** | each f_κ arm's unboosted sibling reproduces its `bench5r __none` baseline | the boost is not the only difference — stop |
| **G3** | every arm compliant (`stop_t=5` or natural end); wall-kills recorded, never dropped | quote no θ from a non-compliant arm |
| **G4** | band entry is **measured in-grid**, not extrapolated | write **"estimated"** in the table — do not extrapolate silently (this is `§18`'s standing flaw) |
| **G5** | both Θ_cum variants reported; frozen-no-root share stated per band-setting arm | do not publish a ranking |
| **G6** | K4's overlapping doses reproduce the `bench6r` ladder to ≤2%, no fire-label flips | the two campaigns are not one measurement — report K4 standalone |

### 5.2 The deliverable

`data/make_bench7_analysis.py` (**not yet written** — its shape depends on what the reduce returns), producing
**the three-way table this program has been missing**:

| knob | bench3 | bench2 | bench1 | spread | measured in-grid? |
|---|---|---|---|---|---|
| f_A | | | | | |
| f_mix | | | | | |
| **f_κ** | | | | | **← the new row** |

with, per G5, **both** Θ_cum variants and the frozen-row share beside every band-setting number.

### 5.3 The decision, and its pre-registered stop

The spread comparison decides which knob behaves most like a single physical constant. Three outcomes,
all pre-committed:

- **f_κ's spread is smallest and its doses are reachable** → f_κ re-enters the head-to-head as a live
  candidate; the two-way published comparison is corrected to three-way.
- **f_κ's spread is smallest but its doses are unreachable** (e.g. entry beyond the condensation boundary)
  → that is a **real result**, not a failure, and it is exactly why G5 exists: the uniformity number may not
  be published without the reachability number beside it.
- **f_κ's spread is largest** → f_κ is closed as a calibration knob, this time on a *measured* basis rather
  than a falsified argument. The `§23` correction stands regardless; it was always about honesty, not
  about promoting f_κ.

**Pre-registered TERMINAL stop:** if no knob holds one constant across the band to within the agreed factor,
the single-constant program stops rather than being re-scoped into a fitted f(n). That is the SC-0 precedent
and it applies here unchanged.

### 5.4 Follow-on work the fresh data unlocks

1. Re-run `make_bench_stale_segments.py` over the new trajectories — the frozen-no-root share feeds G5.
2. Extend `§24` Q1b's back-reaction from its `t ≲ 2.3e-3 Myr` horizon to **full runs**, using K1's `Pb` +
   `bubble_dMdt` columns. This settles P2 on the iterative path rather than at a frozen state.
3. Rebuild `report.html` and re-run the freshness audit; lift the VERIFY tier only for quantities G0
   actually reproduced.

---

## 6. OPEN QUESTIONS (none block the campaign)

| # | question | why it is still open |
|---|---|---|
| **Q1** | f_A's clause-1 grounds — re-derive from the in-ODE structural asymmetry, or withdraw the framing? | `§23` voided the Eq-47-sign leg and `§18` withdrew the measurement leg. The knob's *behaviour* is untouched; its *stated rationale* needs rebuilding on what survives. |
| **Q2** | is `C_f = 1` / `L_leak ≡ 0` expected for the bench configs, or is the leak channel silently disabled? | `§18`, `§20` |
| **Q3** | frozen-row Θ_cum — exclude no-root rows, or carry an uncertainty band? | moves the f_A numbers too. G5 sidesteps it by reporting both variants; a published table still needs the ruling. |
| **Q5** | land the `§16` fallback double-boost fix before the rosette-cf reduction? | `§21` showed 1/36 rosette fm4 fires is bug-dependent |

---

## 7. STATE OF PLAY

| item | state |
|---|---|
| K0 offline re-read | ✅ done (VERIFY tier — pre-cutoff) |
| Campaign design + pre-registration | ✅ frozen |
| G0 (baseline reproduction, pre-run) | ✅ 11/11 |
| G1 (param emit) | ✅ 4/4 — 174 params committed |
| Freshness plumbing + audit | ✅ committed |
| This workstream's docs + `report.html` | ✅ committed |
| **294 arms** | 🔴 **NOT RUN** — needs `ssh helix` |
| `make_bench7_analysis.py` | 🔴 not written (deliberately — shape depends on the reduce) |
| The three-way table | 🔴 **every cell is a prediction** |

**Production is untouched.** Default remains `cooling_boost_mode='none'`, f_κ = 1.0, f_A = 1.0.
