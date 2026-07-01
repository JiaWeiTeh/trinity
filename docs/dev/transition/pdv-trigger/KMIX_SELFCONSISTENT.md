# Self-consistent κ_mix injection — the structural θ response (Rung-B step 2, offline)

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis, not a maintained spec; the code moves faster than these
> notes (paths, line numbers, "what shipped" status drift). **Any agent or person
> reading this: treat it as unverified. Re-check each claim, snippet, and line
> reference against current source before relying on it.**
>
> 🔄 **Living plan — recheck and refine on every visit.** Re-verify the claims and
> line references above against current source; update drift; rethink the strategy
> itself (a better metric, a boundary re-derivation) and note what changed and why
> (date it). Leave it better than you found it. **Keep all banner paragraphs at the top.**
>
> 💾 **Persist diagnostics — commit, don't re-run.** The result is the committed
> `data/make_kmix_selfconsistent.py` → `data/kmix_selfconsistent.csv` +
> `kmix_selfconsistent.png` (reads committed cleanroom trajectories + fixtures, NO sims).
> Reproduce: `python docs/dev/transition/pdv-trigger/data/make_kmix_selfconsistent.py`.
>
> 🔗 **Cross-check the sibling docs — keep the workstream self-consistent.** Siblings:
> `INDEX.md`, `PLAN.md`, `KMIX_PROTOTYPE.md` (step 1, the front estimate), `KMIX_IMPLEMENTATION_SPEC.md`
> (the design this tests + refines), `KMIX_DIFFUSIVITY.md`, `RUNGB_SCOPING.md`. Reconcile any
> number/claim that disagrees; never update one in isolation.

---

## 0. What this is (and the guardrail)

Step 2 of the Rung-B track (`KMIX_IMPLEMENTATION_SPEC.md` §4.3): inject El-Badry's `κ_eff = max(κ_mix,
κ_Spitzer)` into the **real** bubble-structure solver and re-solve, to read what θ the self-consistent profile
yields — the question the front-estimate prototype (`KMIX_PROTOTYPE.md`) could not answer. **No production code
is edited:** the injection is a runtime **monkeypatch** of the conduction in `bubble_luminosity.py`, so the full
production solve `get_bubbleproperties_pure()` (the `v(R1)=0` fsolve for the Weaver dMdt and the whole T-profile)
runs with κ_mix. Builder: `data/make_kmix_selfconsistent.py`. Coverage: the **6 cleanroom configs** (via
`make_da_replay`'s state rebuild on committed trajectories) + the **2 captured fixtures** (stiff 5e9 ≈
`fail_repro`, mild cluster) = 7–8 of the canonical 8 (`small_1e6` control is the gap).

## 1. Correctness gates (both pass — the injection is faithful)

| gate | what | result |
|---|---|---|
| **G1 identity** | at λδv=0 the patched solve == the **unpatched** production solve | **bit-identical (0.0e+00) for all 8** |
| **G2 replay** | at λδv=0 the replayed `bubble_LTotal` == the logged `bubble_Lloss` for the row | **pass (≤7e-7) for all 6 cleanroom** |

G1 is exact because κ_mix is implemented as a **dimensionless multiplier** `max(1, R)` on the existing Spitzer
term (`KMIX_IMPLEMENTATION_SPEC.md` §2) and the Spitzer branch (R≤1, all of it at λδv=0) is the **verbatim
production expression** — so the off-state is the literal same float ops. (First attempt failed G1 at ~1e-6: I
had reordered the κ′-term `2.5·dTdr²/T` as `(2.5/T)·dTdr²`; FP non-associativity, amplified by the stiff ODE,
broke bit-identity. Fixed by emitting the production expression exactly in the Spitzer branch.)

## 2. The physics result (single near-blowout row — **partially superseded by §2b**)

> **⚠️ READ §2b FIRST.** The §2 numbers below are measured at **one near-blowout (max-R2) row** per config.
> The time-resolved follow-up (§2b, `make_kmix_theta_trajectory.py`) shows **that row is the wrong epoch** —
> θ peaks *early* (high Pb) and decays, so blowout is the low-θ *tail*. The §2 "only 1/6 fires / dense
> plateau low" reading **oversold the ceiling for the mid configs**. Finding (1) GO-on-mechanism and finding
> (2) saturation **stand**; finding (3) is corrected in §2b; finding (4) boundary stands.

θ = `bubble_LTotal / Lmech_total` (resolved loss fraction), swept over λδv ∈ {0, 0.01, 0.1, 0.3, 1, 3, 10}
pc·km/s, one representative near-blowout (max-R2) row per config:

| config | n [cm⁻³] | θ(λδv=0) | θ(plateau) | fires (θ≥0.95)? |
|---|---:|---:|---:|:--:|
| large_diffuse_lowsfe | 1e2 | 0.44 | **1.54** | ✅ (overshoots) |
| be_sphere | 1e4 | 0.17 | 0.23 | ❌ |
| midrange_pl0 | 1e4 | 0.17 | 0.24 | ❌ |
| pl2_steep | 1e5 | 0.17 | 0.24 | ❌ |
| simple_cluster | 1e5 | 0.24 | 0.32 | ❌ |
| small_dense_highsfe | 1e6 | 0.31 | 0.35 | ❌ |

**Four findings:**

1. **κ_mix DOES raise the resolved θ in every config, and the solver is stable across the whole sweep
   (6/6).** So the mechanism works *self-consistently in the real solver*, not just as a front estimate — the
   GO from `KMIX_PROTOTYPE.md` is confirmed one level deeper. ✅

2. **θ SATURATES by λδv ≈ 0.01 (6/6) — λδv is NOT a continuous calibration knob.** Because κ_mix swamps
   Spitzer by 10⁵–10⁸ in the cool layer even at λδv ≪ 1 (the prototype's headline), the structure is already
   in the "κ flat in T" regime at the smallest non-zero λδv; raising λδv further does **nothing**. So the plan
   to *calibrate λδv per cloud to hit Lancaster* (`KMIX_DIFFUSIVITY.md` §2) **does not work** — the
   κ_mix-saturated θ is a **fixed output** of each config, not a dial. ⚠️ **This retires the "pin λδv to
   Lancaster" step.**

3. **The saturated θ runs the WRONG way with density.** The diffuse end overshoots (θ=1.54 > 1 ⇒ fires
   immediately), but the mid/dense configs plateau **low** (0.23–0.35), still **far below** the Lancaster
   0.9–0.99 band and **below the 0.95 trigger** — so **κ_mix alone never transitions the dense clouds**. Only
   1/6 reaches the trigger band. κ_mix helps most exactly where help is least needed (diffuse) and least where
   it is needed (dense). ⚠️ **κ_mix is not, by itself, the density-dependent θ knob the goal asked for.**

4. **Boundary finding (refines `KMIX_IMPLEMENTATION_SPEC.md` §3).** Injecting κ_eff into the **boundary** IC
   (site :370) **diverges**: the Spitzer boundary offset `dR2 ∝ C_thermal`, so `κ_eff = C·max(1,R)` scales
   `dR2` by `R(T_init)` — which is ≫1 in the 3×10⁴ K layer — pushing `r2_prime = R2−dR2` past `R1` (invalid
   domain). Patching **both** sites failed at *every* λδv>0; **RHS-only** (site :406; IC kept Spitzer) is
   stable 6/6. So the spec's "site :370 needs κ_eff" is **wrong as stated** — the boundary needs a κ_mix-specific
   re-derivation (the El-Badry layer is not the Spitzer `dR2 ∝ 1/C` closure), not a naive C-scaling. RHS-only is
   the correct scoping choice and likely the correct production choice too. Caught **offline, before production.**

*(The 2 fixtures fail at λδv>0 — the stiff 5e9 is the collapse regime with near-zero θ; consistent with
excluding the heavy cloud, `KMIX_PROTOTYPE.md` §2. They pass G1 identity.)*

## 2a. Why f_κ scaled but κ_mix saturates — the tunable-vs-physical trade (the reconciliation)

A natural objection: *earlier we could scale `f_κ` (`cooling_boost_kappa`) and θ responded as a continuous
knob (the 819-sweep gave `f_κ-to-fire ∝ n^−0.6`; `bubble_LTotal ×1.23–1.38` at f_κ=2). Why does κ_mix not
behave the same?* Because **they are different operations**, and the difference is exactly the saturation:

- **`f_κ` is a uniform scalar on Spitzer:** `κ = f_κ·C_th·T^(5/2)` — it keeps the `T^(5/2)` shape and scales it
  by a *modest* factor. It scales the conduction that is dominated by the **hot interior**, where you sit in
  the **linear** part of the response. So f_κ is a smooth dial — **but it never puts conductivity in the cool
  layer** (Spitzer `T^(5/2) → 0` there), which is why it is *not physical* (it can't represent cool-layer
  mixing) and why it raises dMdt the wrong way.
- **κ_mix is a temperature-independent floor:** `κ_eff = max(κ_mix, κ_Spitzer)`. It lives **entirely in the
  cool layer**, and there it is `10⁵–10⁸ ×` Spitzer the *instant* it is on (even λδv≈0.01, `KMIX_PROTOTYPE.md`
  §2). So the dial (λδv) starts **far past the linear range** — raising it just makes an already-overwhelming
  term more overwhelming; the cool layer is already maximally conducting (isothermal), so θ stops moving.

In one line: **κ_mix is like pinning an *effective f_κ* to a huge value, but only in the cool layer.** f_κ
stays a dial because you keep it modest; κ_mix is *born* in the saturated regime.

|  | tunable? | physical? |
|---|:--:|:--:|
| **f_κ** (Rung A, scalar on Spitzer) | ✅ continuous knob | ❌ a fudge; wrong-sign density dep.; raises dMdt |
| **κ_mix** (Rung B, T-independent floor) | ❌ saturates by λδv≈0.01 | ✅ the faithful cool-layer mixing term |

**The thing we could tune isn't physical; the thing that's physical we can't tune.** The f_κ→κ_mix pivot
traded tunability for faithfulness; this test is where the cost of that trade became visible.

**Consistency, not contradiction:** f_κ *also* saturates — eventually. The sweep's "6/63 low-n/high-sfe cells
never fire even at f_κ=64" (`FINDINGS.md`) is the **same structural θ ceiling** that shows for the **dense
(n≥1e5)** end here (θ_max≲0.5). κ_mix drives every config to its ceiling immediately, exposing it; it didn't
*create* the low dense θ, it *revealed* the ceiling f_κ would also hit. (For mid clouds the "ceiling" is
higher than the blowout row implied — see §2b.)

## 2b. Time-resolved θ — the blowout metric was the wrong epoch (2026-06-30)

The single-row caveat (§2) was the right thing to check, and it mattered. `make_kmix_theta_trajectory.py`
re-solves the structure with κ_mix across **~14 rows of each implicit-phase trajectory** (same RHS-only
injection, λδv=1 saturated vs 0 baseline; reads committed cleanroom trajectories, no sims) →
`data/kmix_theta_trajectory{,_summary}.csv` + `kmix_theta_trajectory.png`.

| config | n | θ blowout (old §2) | θ **max over phase** | θ energy-integral | frac. phase θ≥0.95 | κ_mix solved (early) |
|---|---:|---:|---:|---:|---:|---:|
| large_diffuse | 1e2 | 1.54 | 1.98 | 1.16 | 0.47 | 9/14 (0/4) |
| be_sphere | 1e4 | 0.23 | **1.84** | 0.30 | 0.13 | 8/14 (0/4) |
| midrange_pl0 | 1e4 | 0.24 | **1.14** | 0.20 | 0.13 | 8/14 (0/4) |
| pl2_steep | 1e5 | 0.24 | 0.50 | 0.16 | 0.00 | 8/14 (0/4) |
| simple_cluster | 1e5 | 0.32 | 0.32 | 0.16 | 0.00 | 9/14 (0/4) |
| small_dense | 1e6 | 0.35 | 0.35 | 0.19 | 0.00 | 11/14 (1/4) |

**What this corrects and what it newly exposes:**

1. **The blowout row was the wrong epoch (robust).** θ peaks **early** (high Pb ⇒ large R ⇒ κ_mix most
   dominant) and decays as the bubble expands and Pb drops. Blowout = max R2 = the **late, low-Pb, low-θ
   tail**. So §2 systematically sampled the *minimum* θ. For the mid configs the undersell is severe
   (`be_sphere` 0.23 → trajectory-max **1.84**).
2. **κ_mix is much stronger than §2 implied — but the firing question is NOT cleanly answered.** Where the
   mid-config solves, θ exceeds 0.95 (would fire). But the **early high-Pb rows fail to solve** with the
   hard-max injection (0/4 early rows for every config; the *baseline* off-solve succeeds there, so it is the
   κ_mix injection at large R, not the replay). The decisive epoch is exactly the one we can't yet solve, so
   "mid clouds fire" is **plausible but unconfirmed**.
3. **Dense (n≥1e5) still plateau low where they solve** (θ_max 0.32–0.50, never 0.95) — for the genuinely
   dense end the ceiling looks real. The energy-integral θ_int is modest everywhere except diffuse (0.16–0.30
   for mid/dense), i.e. even the brief mid-config excursions above 0.95 are a small slice of the phase.
4. **Two limitations this run surfaces (next iteration, not fixed here):**
   - **(A) Stability:** the hard `max(κ_mix, κ_Spitzer)` is too stiff at the early high-Pb epochs → the solve
     NaNs exactly where it matters. Needs a **smooth-max** blend `κ_eff = κ_S·(1+R^s)^(1/s)`.
   - **(B) Faithfulness:** κ_mix = (λδv)·n·k_B ∝ **n ∝ 1/T** at fixed Pb, so in the κ_mix-dominated regime
     `(dκ/dT)/κ = −1/T`, **not 0**. The harness (and `KMIX_IMPLEMENTATION_SPEC.md` §3) used 0 ("κ flat in
     T") — a faithfulness bug. It does not move the saturation/wrong-epoch headlines but shifts exact θ; the
     smooth form gives the correct kprime `(1/T)[2.5 − 3.5·R^s/(1+R^s)]` (→ 2.5/T Spitzer, → −1/T κ_mix).

**Net:** κ_mix is a *stronger* and *more promising* correction than the blowout metric showed — plausibly
enough to transition the mid (n~1e4) clouds — but a clean verdict needs a **revised injection (smooth-max +
correct kprime)** that survives the early high-Pb phase. The dense (n≥1e5) low ceiling is the most robust
negative. **This supersedes §2's "only 1/6 fires."**

## 3. What this means for the track (updated after §2b)

- The honest read after the time-resolved check: **κ_mix saturates (not a tunable knob) — that stands — but
  it is a *stronger* correction than the blowout metric showed.** θ peaks early and exceeds 0.95 for the mid
  (n~1e4) clouds where the solve succeeds, so the "only the diffuse end fires" conclusion is **not safe**. What
  *is* robust: λδv is not a dial, and the **dense (n≥1e5)** end stays low (θ_max≲0.5). The "calibrate λδv to
  Lancaster per cloud" hope is still **falsified** (saturation).
- **The clear next step is now a single thing, not a menu** — ✅ route 2 (re-metric) is **done** (§2b) and it
  promoted route 0:
  0. **Revise the injection: smooth-max + correct kprime, then re-run §2b.** `κ_eff = κ_S·(1+R^s)^(1/s)` with
     kprime `(1/T)[2.5 − 3.5·R^s/(1+R^s)]`. This (A) removes the hard-max stiffness that NaNs the early
     high-Pb epochs — the decisive ones for firing — and (B) fixes the κ_mix∝1/T faithfulness bug. **Until the
     early phase solves, the mid-cloud firing question stays open.** Cheap, offline, no sims.
  1. **Then decide combine-vs-cap:** if the revised run confirms mid clouds fire but dense don't, pair a κ_mix
     floor (diffuse/mid) with the gated `theta_target` cap (dense) — each for what it does well.
  2. ✅ **Re-metric — DONE (§2b).** The single-row metric understated θ badly; the trajectory view is the
     measurement to trust.
  3. **Boundary re-derivation** stays a lower-priority lever for the dense ceiling.
- **Production status: still untouched.** Offline measurements that *change the plan*, not the code; no
  registry params added. The maintainer's call: authorize the route-0 revised-injection re-run (recommended —
  it's what stands between "mid clouds plausibly fire" and a confirmed answer), or pick the combine-vs-cap
  direction now on the current evidence.

*Written 2026-06-30 on `feature/PdV-trigger-term-pt2`. No production code touched; monkeypatch-only, no sims.*
