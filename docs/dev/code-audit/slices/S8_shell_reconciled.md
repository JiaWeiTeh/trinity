# S8 shell structure — reconciled

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

**Status (2026-07-29):** 📘 raw agent report — provenance for `FINDINGS.md`; unreconciled and unverified on its own.

**Status (2026-07-30):** 🔀 reconciled slice report. Inputs: `S8_shell_lensA.md` (code, 100 % of the
slice, several claims execution-verified), `S8_shell_lensB.md` (comments/docstrings only),
`S8_shell_lensC.md` (signatures + physics spec, no implementation, literature access blocked).
The reconciler did **not** read source. 69 raw candidates in → **39 reconciled findings out**
(2×S1, 8×S2, 15×S3, 14×S4), with 16 raw candidates dropped or folded and the rationale recorded below.

---

## 1. Coverage

Slice = `trinity/shell_structure/get_shellODE.py` (1–153) and
`trinity/shell_structure/shell_structure.py` (1–473).

| Sub-area | Lens A (does) | Lens B (claims) | Lens C (should) | Reconciled coverage |
|---|---|---|---|---|
| Ionised ODE RHS (`dn/dr`, `dφ/dr`, `dτ/dr`) | full transcription + dimensional check | **3 isolated terms only**, no equation | full independent derivation | ✅ 3-way |
| Neutral ODE RHS | full | fragmentary (no φ step) | full | ✅ 3-way |
| Front locating condition | grid scan `φ ≤ 1e-9`, no root-find | value quoted in the *other* file | root-find `Q = 0` required | ✅ 3-way |
| Front jump (ion→neutral IC) | factor `(µ_n/µ_p)(T_i/T_n)` at :307–308 | "discontinuity", **factor unstated** | pressure continuity ⇒ 200–3667× | ✅ 3-way |
| Strömgren inversion + `min(·, shell_n0)` cap | formula + always-binds proof | formula, citation, **two rationales** | annulus form; `P_HII` must be a *floor* | ✅ 3-way |
| `odeint` call + failure handling | **execution-verified** denormal return | comment admits silent truncation in `simple_cluster` | must raise or flag; fixed-sign bias | ✅ 3-way |
| Mass bookkeeping / loop termination | `any()` over discarded tail; no-op line | "neutral iff φ-depleted and mass remains" | I1 mass closure, event-terminated free boundary | ✅ 3-way |
| Photon bookkeeping / absorbed fractions | φ-sink re-integration, ratio normalisation | 4 fields, normalisation unstated | I2 `f_gas+f_dust+f_esc = 1` | ✅ 3-way |
| Units / dimensions | full table, **no imbalance found** | docstring units; `τ_IR/κ_IR` unitless | cgs↔AU mandate + numeric anchors | ⚠️ symbols checked, **magnitudes not** |
| Radiation-force form & EOM consistency | code has no inertial/gravity term | **no prose at all** | S8.1/S8.2, invariant I10 | ⚠️ A+C only (B silent) |
| Dissolution / state flags | strict `<`, dissolved arm constants | two flags, one-step lag | I12 (`shell_nMax` from converged profile) | ✅ 3-way |
| `f_cover` | ionised `dτ/dr` only, hardwired 1 | documented **and** two TODOs | must enter exactly once, in the column | ✅ 3-way |
| Grid resolution | literals `1e3`/`5e3`/`1 pc` | nothing stated | T6/C-19: front must be root-found | ⚠️ A+C only |
| Purity / statelessness | no mutation reported | contract asserted 5× | — | ⚠️ B only, untested |
| Return-array aliasing | 3 fields share one buffer | — | — | ⚠️ A only |
| `P_HII` consumer, IR/LyC force bookkeeping, Z-scaling | **outside slice** | outside slice | C-15/16/18/25 reach for it | 🚩 hand-off to force-budget slice |

Coverage gaps that matter: (i) no lens can see the *numeric values* of `dust_sigma`, `caseB_alpha`,
`chi_e_shell`, `mu_convert`, `TShell_neu`, `nCore` — every magnitude-level check below is
therefore deferred; (ii) no lens can see the `P_HII` consumer, which is the one link needed to
close S8-R-02.

---

## 2. Integration failure — the primary reconciliation

### 2.1 What each lens says

**Lens A (execution-verified, strongest evidence in this audit).** `shell_structure.py:165–171`
and `:324–329` call `scipy.integrate.odeint(..., mxstep=_SHELL_ODE_MXSTEP)` with **no
`full_output`, no `try/except`, no `warnings.catch_warnings`**, then immediately consume every
column. A verified in-environment (scipy 1.17.1) that a failed `odeint` with `full_output=0`
emits only an `ODEintWarning` and returns a **full-length** array whose rows past the failure
point are **uninitialised heap memory** — denormals of order `1e-310`. Those denormals satisfy
`phiCondition = phiShell_arr <= 1e-9` (`:182`), so `idx` lands on the first garbage row and
`is_phiDepleted` is set `True`: *a solver failure is reinterpreted as "the ionisation front is
here."* `n_IF`, `n_IF_ODE`, `R_IF`, `f_esc_ion` → `shell_fAbsorbedIon` →
`shell_fAbsorbedWeightedTotal`, `n_IF_Str`, `shell_grav_r/_phi/_force_m`, `shell_fIonisedDust`,
`shell_thickness`, `shell_nMax`, `shell_tauKappaRatio`, `rShell`, `shell_r_arr`/`shell_n_arr`
and `diss_condition_met` are all then derived from that memory. `has_neutral` becomes `True`
with `nShell0 ≈ 1e-310`, so the neutral loop's mass condition can never be met and it marches
outward forever.

**Lens C (derived from the interface, no code seen).** S8-C-07/08/21: `odeint` does not raise on
`mxstep` exhaustion; a `_SHELL_ODE_MXSTEP` constant only protects anything if `full_output=True`
and `infodict`/`ier` are inspected. A partial profile must never be consumed; `ShellProperties`
must carry a validity flag propagated to the snapshot. And crucially: **every truncation error
carries a fixed sign** — τ under ⇒ `f_abs` under ⇒ `F_rad` too small; `f_esc` over;
`shell_nMax` under ⇒ spurious dissolution; `M(r_last) < M_sh` ⇒ silent mass loss. Systematic
bias, not noise, so it does not average out over a sweep.

**Lens B (prose only) — what the code claims about error handling.** The *only* error-handling
prose in the slice is `shell_structure.py:28–34`, and it is an admission: `odeint`'s default
`mxstep=500` "is exhausted in the degenerate code-unit-overflow regime (**simple_cluster**),
where it emits 'Excess work done on this call' and **silently truncates the shell
integration**." The documented mitigation is raising the ceiling to `50k` — "Robustness fix
only -- same LSODA solver" — and the bit-identical equivalence claim is explicitly scoped to
"where the ceiling was never hit". B records that **no prose anywhere claims the return code or
`infodict` is inspected**, and nothing says what to do if the raised ceiling is also exhausted.
B also catalogues five further *silent* corrections (n-cap, φ-clamp, two τ-underflow guards, the
sub-threshold-φ guard) — none of which is documented as counted, logged, or surfaced.

### 2.2 Same mechanism or two?

**One mechanism, two resolutions of it.** All three describe the identical code path: an
`odeint` return consumed without any status inspection. They differ on *what is in the buffer*,
and A wins on evidence:

- C (and B's comment) model the failure as "returns the **partial trajectory** / silently
  truncates" — i.e. valid rows up to the failure, then either a short array or the last good row.
- A **executed it**: the array is full length and the tail is *uninitialised memory*, not the
  last good row. This is a materially worse failure than either doc or spec assumed, because the
  values are not merely stale — they are arbitrary, run-to-run non-deterministic, and happen to
  fall in the exact numeric range (`~1e-310 ≤ 1e-9`) that the code's own front test accepts.

**Consequence for C's fixed-sign claim — partially retracted.** C's sign analysis assumes the
last *valid* row is consumed. Under A's observed mechanism the signs are mixed:

| quantity | C's predicted bias (partial trajectory) | bias under A's observed denormal tail | agree? |
|---|---|---|---|
| `τ_end`, `f_abs` | under | under (truncated early) | ✅ |
| `f_esc` | **over** | **under** — `f_esc = max(0, φ≈1e-310) → 0`, so `f_abs,ion → 1` | ❌ |
| `shell_nMax` | under | under (max over the truncated valid rows) | ✅ |
| mass closure | `M < M_sh` | `M < M_sh`, but flagged swept ⇒ undetectable | ✅ |
| overall outcome | biased published numbers | **hang** in the neutral loop (A §4.5) | ❌ |

So: C's structural argument (a truncated outward integration biases everything in a fixed
direction, and biases do not average out over a sweep) **survives and is the right reason to
treat this as S1**; C's specific prediction that `f_esc` is *over*-reported does **not** survive
contact with A's mechanism, which drives `f_esc` to exactly 0 and `f_abs,ion` to 1. Both lenses
are right that the error is signed; they disagree on the sign of one headline output. Retain the
disagreement — it is the single most useful thing a repro would settle.

### 2.3 Reachability — S1 or S2

Evidence for the path being live:

1. **Demonstrated at the previous default.** B's verbatim comment names `simple_cluster` — the
   tracked quickstart/baseline config, per project `CLAUDE.md` the documented single-run example —
   as the regime where `mxstep=500` was exhausted and the integration silently truncated. That is
   the code's own admission that the failure *has already happened, in the flagship config*.
2. **The stiffness that caused it is still there.** Raising `mxstep` to 50 000 changes *when* the
   ceiling fires, not *whether* failure is detected (A, C, and B's own "Robustness fix only"
   wording all agree). The `+n²` pole past the front, the `1e120` state cap and the φ=0 kink
   (S8-R-12, S8-R-26) still degrade LSODA's error control in exactly the region the profile is
   read from.
3. **Detection is unconditionally absent.** A's "no `full_output`, no `try/except`" is a
   structural fact of the current source, not a regime-dependent one. Whatever the trigger
   probability, the *consequence* is undefined-behaviour-class.

Evidence against:

4. No lens demonstrates a `mxstep=50 000` exhaustion in any config run today. B's comment claims
   the raised ceiling "lets the solve complete" across 6 configs, and A's repro used a synthetic
   ODE with `mxstep=50`, not trinity.

**Verdict: S1**, on the grounds that the consequence is consumption of uninitialised memory as
physics with zero detection, and the config family that triggered it at 500 is the tracked
baseline. **Explicit demotion trigger:** if instrumentation (`full_output=1`, log `infodict['nst']`
against `mxstep`) shows the step budget is never approached — say, max steps < 10 % of 50 000 —
across `param/simple_cluster.param` and both
`docs/dev/performance/f1edge_{lowdens,hidens}*.param`, demote to **S2** (latent, guarded by
margin). Until that measurement exists, S1 stands. Note this instrumentation is also *the fix*,
so measuring and fixing are the same edit.

### 2.4 Smallest test in the real `pytest` suite

One new case, in the existing suite, physically plausible inputs (project convention), no new
dependency:

```python
# test/test_shell_ode_failure.py
import numpy as np, pytest
import trinity.shell_structure.shell_structure as ss

@pytest.mark.timeout(60)            # the failure mode A predicts is a hang, not a wrong number
def test_shell_integration_failure_is_detected(monkeypatch, simple_cluster_params):
    monkeypatch.setattr(ss, "_SHELL_ODE_MXSTEP", 5)     # force the documented failure
    props = ss.shell_structure_pure(simple_cluster_params)
    tiny = np.finfo(float).tiny                          # 2.2e-308
    prof = np.asarray(props.shell_n_arr)
    assert np.all((prof == 0.0) | (np.abs(prof) > tiny)) # subnormal ⇒ uninitialised memory
    assert props.n_IF == 0.0 or abs(props.n_IF) > tiny
```

Why this is the minimum: a physical shell density in code units is `~1e55–1e65 pc⁻³`, so a
**subnormal** anywhere in the returned profile cannot be physics — it is a positive signature of
the un-integrated tail, and it needs no reference trajectory, no separate process and no
long run. The `timeout` marker is load-bearing: if A's neutral-loop non-termination (S8-R-11)
fires first, the test fails by timing out, which is still the correct verdict. Once the fix
lands, flip the body to `with pytest.raises(...)` (or assert the new validity flag of S8-R-07).
Add A's one-line scipy-contract check as a second case so a future scipy release that changes
the failure return is caught:
`odeint(lambda y,t:[y[0]**2*1e6],[1.0],np.linspace(0,10,20),mxstep=50)`.

---

## 3. The always-binding clamp (`min(n_IF_Str, shell_n0)`)

**A's argument, restated.** Every term of `dn/dr` in the implemented ionised RHS is non-negative
(σ_d, L_n, L_i, χ_e, α_B, Q_i, c > 0; `e^{-τ} ≥ 0`; φ clamped ≥ 0), so `n(r) ≥ n0`, increasing.
The raw inversion is `n_raw² = Q_abs/(χ_e α_B ΔV_geo)` over the *same* interval `[R2, R_IF]` the
ODE integrated, while hydrogen alone consumed `Q_H = χ_e α_B ⟨n²⟩ ΔV_geo` with `⟨n²⟩ ≥ n0²`.
Since `Q_abs = Q_H + Q_dust ≥ Q_H`, `n_raw ≥ n0` ⇒ `min(n_raw, shell_n0) ≡ shell_n0`.

**The algebra holds.** The endpoint used for `ΔV` and the endpoint of the φ integration are the
same (`R_IF = rShell_arr_ion[-1]`), so grid overshoot cannot break the inequality; equality
occurs only in the exactly-uniform limit, where the clamp is harmless either way. Both of B's
documented regimes (`is_phiDepleted` True/False) reduce to one expression in code (A: same
`R_IF`, same `f_esc`), so the proof covers both.

**But the premise is conditional — demote "provably always" to "always, as implemented".**
Lens C's derivation of the same equation (S8.1) has `dP/dr = −ρ(a_sh + GM/r²) + f_rad`: the
inertial and gravity terms are **negative**. C's invariant I5 states `dP/dr ≥ 0` *only* "when
`f_rad ≥ 0` **and inertia is neglected**". A's transcription shows the code has no inertial and
no gravity term in the RHS, so `dn/dr ≥ 0` holds identically **in the current code** — but it is
a consequence of an omission (S8-R-05), not of physics. The moment the missing `−ρ a_sh` term is
added, `n(r)` can fall below `n0` and the clamp stops always binding. So: A's conclusion stands
for today's code; its "provably" is doing work that the omitted physics, not the mathematics,
is paying for. Recorded as such in S8-R-04, with S8-R-05 as its stated premise.

**And the clamp is the load-bearing half of a much bigger finding.** B documents the cap twice,
with two rationales: `:239` "`n_IF_Str ≤ shell_n0` (pressure equilibrium for thin skins)" and
`:250` "**P_HII cannot exceed P_b**". A shows `shell_n0` is itself back-solved from a pressure
(`:124–125`, `n = µ_p P/(µ_H k_B T_i)` — the identical combination that prefixes `dn/dr`), and
B documents `n_IF_Str` as the "**sole source of P_HII**" (`:77`). C, independently and without
seeing any of this, derives (S8-C-18 / T10) that `P_HII` computed from a density back-solved
from the driving pressure makes `max(P_b, P_HII)` **an exact no-op**, and that the physical
content of `P_HII` is a *floor* that must be able to exceed `P_b` — the opposite of a ceiling at
`P_b`. Three lenses, three routes, one conclusion → **S8-R-02, S1**. B's "contradiction A"
(S8-B-02, two cap descriptions) dissolves: capping `n` at `shell_n0` *is* capping `P_HII` at the
pressure that seeded `shell_n0`; the two comments describe the same operation, and that
operation is the defect.

---

## 4. The ionisation front

**C's derived requirement.** Weak-D front in a quasi-static shell ⇒ subsonic mass flux ⇒ the
momentum jump reduces to **pressure** continuity, so density jumps *up* by
`(ψ_ion T_ion)/(ψ_neu T_neu)` = 200 (atomic, 100 K) to 3667 (molecular, 10 K) `[computed]`.
Carrying `n` continuously while switching only `T`/`ψ` would impose an R-type front and drop the
internal pressure by 200–3700×.

**What the code actually does — C's S1 does not survive.** A reports the neutral loop is seeded
with `nShell0 ← nShell_ion · (µ_n/µ_p)(T_i/T_n)` (`:307–308`). In the code's own convention
(`P = n µ_H k_B T/µ`, from A's :124–125), pressure continuity is exactly
`n_neu = n_ion (µ_n/µ_p)(T_i/T_n)`, and `µ_n/µ_p = ψ_ion/ψ_neu`. **The code implements precisely
C's required jump, with precisely C's factor.** S8-C-06 is therefore demoted from S1 to two much
smaller items: the factor is undocumented (B's `:306` "Temperature/density discontinuity at
boundary", no number → S8-R-35, S4) and the neutral layer is *always atomic* — no molecular
branch appears anywhere in A's transcription — so a 10 K molecular neutral shell gets a jump
`~1.8×` too small, propagating into `shell_nMax`, `τ_UV` and `τ_IR` (S8-R-14, S3). This is the
clearest example in the slice of a derived S1 that contact with the code demotes.

**The sharp-front validity finding survives, with caveats.** C derives
`ℓ_IF/ΔR_ion = 4π R2² α_B χ_e n_H/(σ_HI Q_i)` = 0.14 at `n_H = 1e4` and **1.4 at `n_H = 1e5`**,
i.e. the boolean `is_ionised` (x ≡ 1 inside / 0 outside) approximation breaks at
`n_H ≳ 1e5 cm⁻³` — a density C says TRINITY's own `nCore = 1e5` default reaches, and which a
*compressed* shell exceeds by a large factor. Reconciled status:

- **Corroborated leg:** no validity check exists. A covered 100 % of both files and reports no
  ionisation-fraction variable, no `x(r)`, no warning, no flag; B reports no documented validity
  range for the `x = 1` assumption anywhere (only "shell HII is singly ionised"). Two lenses,
  positive evidence, not silence.
- **Single-lens leg:** the numeric threshold. It rests on C's `[recalled]` `σ_HI = 6.3e-18 cm²`
  and `α_B = 2.59e-13 cm³ s⁻¹` and on an example (`Q_i = 1e49`, `R2 = 5 pc`) not drawn from a
  TRINITY config. The reachability claim (`nCore = 1e5` default) cites the physics spec, not the
  schema in `trinity/_input/`, and **no lens verified it**.
- **Verdict S2, confidence medium.** High value if the reachability leg confirms; the two checks
  that settle it are cheap (read the `nCore` default from the schema; evaluate the ratio from
  stored snapshots). The bias direction is fixed and grows with density — worst exactly in the
  dense-cloud runs the paper features — so it cannot be argued away as scatter.

**Also folded in here:** C's T2/S8-C-13 (bare `R_St` must never be used as a layer thickness).
A shows `max_shellRadius = R_St(n0) + r_start` is used **only** to size slices, never as the
front location — so C's S1 does not survive as a physics error. But it survives as the *cause*
of a resolution problem: `R_St` over-states the annulus thickness by `3R2²/R_St²` (~3700× in
C's example), and the ionised grid step is `R_St/1e4`, so the number of grid points spanning the
ionised layer is `≈ 1e4 · R_St²/(3 R2²)` — **≈ 2.6 points** for C's `R_St/R2 = 0.028`. The front
is located by scanning that grid (A: no root-find), so `R_IF`, `f_esc`, `n_IF` and `f_dust` are
quantised at the grid spacing → **S8-R-06, S2, A+C corroborated**.

---

## 5. Mass and photon bookkeeping

**Mass (corroborated, S8-R-03, S2).** A: `massCondition`/`phiCondition` are computed over the
whole slice; `idx` is the first index of their union; `mShell_arr_cum[idx+1:] = 0.0` (`:190`) is
a **no-op** (the bool array was already snapshotted at `:181` and every later read is `[:idx]`
or `[idx]`); then `:191–192` `any()` both conditions over the **entire slice, tail included**. If
φ depletes at index 10 while the unphysical continuation sweeps the target mass by index 500,
`is_allMassSwept` is set `True`, `has_neutral` becomes `False` (`:221`) and the neutral region is
**never integrated** although the shell is φ-depleted with mass left to sweep. C, from the
interface alone, requires exactly the invariant this violates: I1 `∫4πr²ρ dr = M_sh` with the
residual **checked, not assumed** (S8-C-09). Two lenses, two methods, one defect: the code
believes the mass closed (the flag says so) while the collected profile is short. Nothing
anywhere computes a mass residual. The fix A proposes (`massCondition[idx]`, `phiCondition[idx]`)
is presumably what `:190` was meant to achieve — which is also why `:190` is dead rather than
merely redundant.

C's broader S8-C-09 framing ("integrating a guessed interval and truncating") is **demoted**: the
code *is* event-terminated on the mass condition and marches slices until it is met, so closure
is enforced to one grid cell — except on the S8-R-03 path, where it is not enforced at all and
not checked.

**Photons (C's S1 dropped).** C-05 demands `dQ/dr = −4πr²[α_B χ_e n² + σ_d n Φ]` with
`Φ = Q/(4πr²)`, warning that a Φ-form needs the `−(2/r)Φ` geometric term and that mixing forms
is a `4πr²` error. A's transcription:
`dφ/dr = −4πr²χ_e α_B n²/Q_i − n σ_d φ`. With `φ ≡ Q/Q_i` this is C's Q-form **exactly** —
photon-number-conserving, no geometric term needed, no mixing. **Dropped: the code already does
the thing C asked for.** What survives is a normalisation ambiguity: A shows
`shell_fIonisedDust = φ_dust/(φ_dust + φ_H)` is a fraction **of the absorbed ionising photons**,
while B records it documented as "Fraction of ionizing radiation absorbed by dust" — read
naturally as a fraction of the total. A consumer picking the wrong reading is off by exactly
`f_abs,ion` (S8-R-22, S3, A≠B). Note also that the two sink integrals are re-accumulated with a
left-endpoint rectangle rule over a steeply-varying integrand — first-order and independent of
the LSODA-integrated φ — so `f_gas + f_dust + f_esc = 1` (C's I2) will **not** hold to machine
precision even though the ODE form is right; the ratio normalisation hides it.

---

## 6. Divergence table (all remaining items)

Class key: **A≠B** doc-drift · **A≠C** physics · **B≠C** mis-cited literature ·
**scope-creep** = A=B=C-unsanctioned (spec reaches past the slice) · **none** = the lenses agree.

| id | topic | class | status | verdict |
|---|---|---|---|---|
| R-01 | `odeint` return consumed unchecked | AC (buffer contents & bias sign) | corroborated | **kept S1**; A's execution evidence supersedes C/B's "partial trajectory" model |
| R-02 | `n_IF_Str ≡ shell_n0` ⇒ `P_HII` capped at `P_b` ⇒ `max()` inert | BC (doc says ceiling, spec says floor) | corroborated (3 legs) | **kept S1**; highest-value derived finding |
| R-03 | flags `any()`-ed over the discarded tail | none (A mechanism, C invariant) | corroborated | kept S2 |
| R-04 | `min(n_IF_Str, shell_n0)` always binds | none | single-lens (A) | kept S2, "provably" **demoted to conditional** on R-05 |
| R-05 | shell ODE omits `−ρ a_sh` and gravity | AC | corroborated | **demoted C S1→S2** (global EOM lives elsewhere) |
| R-06 | ionised layer under-resolved; front grid-scanned | AC | corroborated | kept S2; absorbs C-13 (`R_St`) and A-16 |
| R-07 | no validity/convergence flag on `ShellProperties` | none | corroborated (A,B,C) | kept S2 |
| R-08 | sharp-front approx. unflagged at `n_H ≳ 1e5` | AC | corroborated (gap) / single-lens (threshold) | kept S2, constants `[recalled]` |
| R-09 | `simple_cluster` = the degenerate overflow regime | none | corroborated | kept S2; reachability evidence for R-01 |
| R-10 | neutral `sliceSize ≤ 0` | none | single-lens (A) | kept S2, medium |
| R-11 | neutral loop unbounded; `tau_max` dead | none | single-lens (A) | kept S3 |
| R-12 | φ clamped in the RHS only, no equilibrium at 0 | none | corroborated (A,B,C) | kept S3; C's "clamped state ⇒ spurious front" mechanism is *not* what the code does |
| R-13 | `f_cover`: ionised branch only, hardwired 1, wrong form | AC + AB | corroborated | kept S3; B's "silently ignored" **refuted** by A |
| R-14 | neutral layer always atomic (no `ψ_mol`) | AC | corroborated | kept S3 (survivor of C-06) |
| R-15 | dissolution state machine (lag + strict `<`) | AB | corroborated | kept S3 |
| R-16 | dissolved shell still pays the full integration | none | single-lens (A) | kept S3 |
| R-17 | `Q_i = 0` ⇒ `ZeroDivisionError` before the ODE | none | single-lens (A) | kept S3 |
| R-18 | shell ODE never written in prose, no citation | AB | corroborated | kept S3 |
| R-19 | "Lancaster+2025, generalised" unresolvable | BC | single-lens (B) | kept S3 — **literature settles** |
| R-20 | `tau_kappa_IR` column, units undocumented | AB | corroborated | kept S3 |
| R-21 | cgs↔AU constant *magnitudes* unverifiable in-slice | none | single-lens (C), narrowed by A | kept S3 — anchor tests |
| R-22 | `shell_fIonisedDust` normalisation | AB | corroborated | kept S3 |
| R-23 | `shell_ion_idx = -1` sentinel is a legal index | none | single-lens (B) | kept S3 |
| R-24 | `ΔR/R2 ≪ 1` never checked | none | single-lens (C) | **demoted C S2→S3** (the ODE itself is spherical, not thin-shell) |
| R-25 | purity contract asserted 5×, never tested | none | single-lens (B) | kept S3 |
| R-26 | `_NSHELL_MAX = 1e120` clamp in the RHS | AB | corroborated | kept S4 |
| R-27 | `τ > 500 ⇒ e^{-τ} := 0` discontinuity | AB | corroborated | kept S4; B-13's "branches may differ" **refuted** by A (both 500) |
| R-28 | `mShell_arr_cum[idx+1:] = 0.0` no-op | none | single-lens (A) | kept S4 |
| R-29 | `n_IF_ODE` duplicates `n_IF` | AB | corroborated | kept S4; B-07's divergence worry **refuted** (both 0.0 when dissolved) |
| R-30 | unused mass arrays; element 0 mixes cum/diff | none | single-lens (A) | kept S4 |
| R-31 | returned arrays alias one buffer | none | single-lens (A) | kept S4 |
| R-32 | bare dimensioned literal `1` (= 1 pc) | none | single-lens (A) | kept S4 |
| R-33 | `(Li + Ln)` divisor unguarded | none | single-lens (A) | kept S4 |
| R-34 | docstring/debris hygiene bundle | AB | corroborated | kept S4 (B-04, B-18, B-10, typo) |
| R-35 | front jump factor implemented but undocumented | AB | corroborated | kept S4 (survivor of C-06) |
| R-36 | `rShell` frozen when dissolved, bug-compat | none | single-lens (B) | kept S4 |
| R-37 | Lyman-α radiation pressure neglected silently | none | single-lens (C) | kept S4, `[recalled]` |
| R-38 | IR form, LyC double-count, Z-scaling | scope-creep | single-lens (C) | kept S4 as hand-off to the force-budget slice |
| R-39 | φ threshold documented in the wrong file | AB | corroborated | kept S4 |

### Dropped (raw candidate → why)

| raw | why dropped / folded |
|---|---|
| S8-C-02 (`f_rad` form, S3) | A's RHS is `(n σ_d/c)·L/(4πr²)` = `(dτ/dr)F/c` — exactly C's required form, no `2/r` term. **No defect.** |
| S8-C-03 (`α_B χ_e n_H²`, S1) | A shows `chi_e * caseB_alpha * nShell**2` with `nShell` the H-nuclei density and a *separate* `µ_p` for pressure — C's SPEC-092 item 1 distinction handled correctly. Magnitudes deferred to R-21. |
| S8-C-04 (case B, S2) | The parameter is literally `caseB_alpha`. Value deferred to R-21. |
| S8-C-05 (photon conservation, S1) | The code integrates `φ = Q/Q_i` with both true sinks — C's own preferred Q-form. **No defect.** |
| S8-C-06 (density jump, S1) | Jump is implemented with C's exact factor → split into R-14 (S3) + R-35 (S4). |
| S8-C-09 (mass closure, S2) | Loop is event-terminated on mass; folded into R-03 as the invariant that would catch it. |
| S8-C-10 (`f_esc` exact, S2) | `f_esc = max(0, φ(R_IF))` — never floored, clipped or fitted; the `max(0,·)` concern is R-12. |
| S8-C-11 / C-12 (units, S1×2) | A's dimensional table finds **no imbalance**; σ_d multiplies `n`, not ρ. Narrowed to R-21 (magnitudes only). |
| S8-C-13 (`R_St` as thickness, S1) | `R_St` sizes slices only, never the front → folded into R-06 as the *cause* of the resolution problem. |
| S8-C-20 (switch `ψ` and `T` together, S2) | Both `µ` and `T` switch (`µ_p→µ_n`, `T_i→T_n`) → only the atomic/molecular choice survives, as R-14. |
| S8-B-01 (`ΔV` volume vs `R³−R2³`, S2) | A's code reading gives `3 Q/(4π χ_e α_B (R_IF³−R2³))` — prefactor and substitution are mutually consistent. **No factor error**; naming nit only. |
| S8-B-02 (two cap rationales, S3) | Not a contradiction: `shell_n0` is back-solved from a pressure, so a density cap at `shell_n0` *is* a pressure cap. Folded into R-02 — the agreement is the defect. |
| S8-B-03 (`f_cover` inert, S2) | Refuted in detail: `f_cover` is plumbed and *is* applied in the ionised `dτ/dr`. Folded into R-13. |
| S8-B-07 (`n_IF_ODE` may diverge, S3) | Refuted: both are set to `0.0` on the dissolved arm. Folded into R-29 as pure redundancy. |
| S8-B-13 (τ guards may differ, S4) | Refuted: both branches use `500`. Folded into R-27. |
| S8-B-22 (`f_esc ≈ 0` continuity, S4) | Refuted: one expression, one `R_IF`, both "regimes" — continuity is trivial. |

### What the actual literature would settle

C flagged its constants `[recalled]` vs `[computed]` because literature access was blocked. These
items cannot be closed without the sources:

1. **R-19 / Lancaster+2025** — whether the `(1 − f_esc)` factor, the `R³ − R2³` annulus form and
   above all the `n ≤ shell_n0` cap come from the paper or are local inventions, and what
   "generalised" changed. This is the *only* citation in the slice and it directly gates R-02.
2. **R-08 threshold** — `σ_HI = 6.3×10⁻¹⁸ cm²` `[recalled, high]` and `α_B(10⁴ K) =
   2.59×10⁻¹³ cm³ s⁻¹` `[recalled, high]`, which set the `n_H ≳ 1e5` breakdown density.
3. **R-21 anchors** — `σ_d = 1.5×10⁻²¹ cm²/H`, `κ_IR ≈ 4 cm² g⁻¹`, `α_A/α_B = 1.61`
   `[recalled, medium]`, `⟨hν_i⟩ ≈ 15–18 eV` `[recalled, medium]`.
4. **R-37** — `≈0.68` Lyα photons per case-B recombination `[recalled, medium]` and the
   resonant-trapping force multiplier.

Everything else C tagged `[computed]` (the 200/3667 jump ratios, `ΔR_ion = R_St³/(3R2²)`, the
`κ_UV/κ_IR ≈ 160` anchor) is re-derivable arithmetic and needs no literature — only the input
constants above.

---

## 7. Merged ranked findings

```json
[
  {
    "id": "S8-R-01",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 165,
    "class": "silent-failure",
    "severity": "S1",
    "claim": "scipy.integrate.odeint is called at :165-171 and :324-329 with no full_output, no try/except and no warnings handling, and every column of the return is consumed unconditionally. On integration failure scipy returns a FULL-LENGTH array whose un-integrated rows are uninitialised heap memory (denormals ~1e-310); those denormals satisfy the front test phiShell_arr <= 1e-9 at :182, so a solver failure is silently reinterpreted as 'the ionisation front is here'.",
    "evidence": "EXECUTION-VERIFIED (Lens A, scipy 1.17.1): odeint(lambda y,t:[y[0]**2*1e6],[1.0],np.linspace(0,10,20),mxstep=50) emits only an ODEintWarning and returns array([1.0, 4.873, 6.90167858e-310, 6.90167857e-310, 1.387e-315, ...]). Lens B: the code's own comment at :28-34 admits odeint 'emits Excess work done on this call and silently truncates the shell integration' in the simple_cluster regime, that the mitigation was raising mxstep 500 -> 50000 ('Robustness fix only -- same LSODA solver'), and that no prose anywhere claims the return status is inspected. Lens C, from the interface alone: odeint does not raise on mxstep exhaustion, so _SHELL_ODE_MXSTEP protects nothing unless full_output/ier or sol.status is checked; a partial profile must never be consumed because every truncation error carries a FIXED SIGN and therefore biases a sweep rather than adding noise.",
    "expected": "Pass full_output=1 and inspect infodict/ier (or catch ODEintWarning as an error), then raise with a recorded termination outcome, or fall back to a documented closed form AND set a validity flag on ShellProperties that is propagated to the snapshot (see S8-R-07). Never consume the returned buffer unchecked.",
    "failure_scenario": "LSODA hits mxstep in the stiff ionised region. idx lands on the first garbage row and is_phiDepleted becomes True; n_IF, n_IF_ODE (:224-225), R_IF (:226), f_esc_ion (:229) -> shell_fAbsorbedIon (:398) -> shell_fAbsorbedWeightedTotal (:400), n_IF_Str (:246), shell_grav_r/_phi/_force_m (:262-273), shell_fIonisedDust (:277-288), shell_thickness (:392), shell_nMax (:394), shell_tauKappaRatio (:395), rShell (:402), shell_r_arr/shell_n_arr (:410-414) and diss_condition_met (:446) are all derived from uninitialised memory. has_neutral then becomes True with nShell0 ~ 1e-310, so the neutral loop at :316 can never satisfy massCondition and does not terminate. NOTE the reconciled sign disagreement: Lens C predicts f_esc OVER-reported (last valid row consumed); Lens A's observed denormal tail drives f_esc -> exactly 0 and f_abs,ion -> 1, i.e. UNDER-reported. tau/f_abs under-estimated, shell_nMax under-estimated and mass silently lost are agreed by both.",
    "repro": "python -c \"import numpy as np,scipy.integrate; print(scipy.integrate.odeint(lambda y,t:[y[0]**2*1e6],[1.0],np.linspace(0,10,20),mxstep=50).ravel())\"  ;  then test/test_shell_ode_failure.py: monkeypatch _SHELL_ODE_MXSTEP=5, run shell_structure_pure on simple_cluster params under a 60 s timeout, assert no returned float is subnormal (abs(x) > np.finfo(float).tiny or x == 0).",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S8-A-01", "S8-B-05", "S8-C-07", "S8-C-08"]
  },
  {
    "id": "S8-R-02",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 251,
    "class": "other",
    "severity": "S1",
    "claim": "n_IF_Str -- documented as the SOLE source of P_HII -- is identically shell_n0, and shell_n0 is itself back-solved from the shell's inner-face pressure. The clamp is therefore documented and implemented as a ceiling 'P_HII cannot exceed P_b', which makes any downstream max(P_b, P_HII) an exact no-op, whereas the physics requires P_HII to be a FLOOR that can exceed P_b.",
    "evidence": "Lens A (read-only proof): every term of the ionised dn/dr is non-negative, so n(r) >= n0; the raw inversion n_raw^2 = Q_abs/(chi_e alpha_B dV_geo) over the same interval the ODE integrated satisfies Q_abs >= Q_gas = chi_e alpha_B <n^2> dV_geo with <n^2> >= n0^2, hence n_raw >= n0 and min(n_raw, shell_n0) == shell_n0 always. A also shows shell_n0 is computed at :124-125 as n = mu_p P/(mu_H k_B T_i), i.e. back-solved from a pressure. Lens B: :77 'Stroemgren ionization balance density (Lancaster+2025), sole source of P_HII'; :239 'n_IF_Str <= shell_n0 (pressure equilibrium for thin skins)'; :250 'Cap: thin ionised skin -> P_HII cannot exceed P_b'. Lens C (independent, interface only): S8-C-18/T10 -- if P_HII is computed from a density back-solved from P_drive then P_HII == P_drive identically and max(P_b, P_HII) is inert; the non-circular content of P_HII is a floor set by the Stroemgren balance over the available shell mass.",
    "expected": "P_HII derived from an independently pinned geometry (Q_i, M_sh, R2) and demonstrably able to exceed P_b; the cap, if it is kept at all, reported as a diagnostic rather than applied silently -- as written n_IF_Str carries no information beyond shell_n0, which is already returned separately.",
    "failure_scenario": "TRINITY's headline pressure term is inert: the P_HII branch of max(P_b, P_HII) never activates, yet a P_HII column is still reported and a four-line Stroemgren inversion is computed and discarded on every timestep.",
    "repro": "Scan dictionary.jsonl from param/simple_cluster.param and both docs/dev/performance/f1edge_* configs for ANY snapshot with P_HII > Pb; and assert n_IF_Str == shell_n0 exactly on every snapshot. If both hold, the degeneracy is confirmed. Then confirm which pressure feeds shell_structure.py:124-125.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "BC",
    "status": "corroborated",
    "source_ids": ["S8-A-02", "S8-B-02", "S8-B-09", "S8-C-18"]
  },
  {
    "id": "S8-R-03",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 191,
    "class": "state",
    "severity": "S2",
    "claim": "is_allMassSwept and is_phiDepleted are any()-ed over the ENTIRE integrated slice including grid points past idx, which lie beyond the physical end of the shell, so a termination flag can be set by the discarded tail; the neutral region is then skipped while shell mass remains and the mass-closure residual is never checked.",
    "evidence": "Lens A: :181-192 compute massCondition and phiCondition over the full slice, take idx as the first index of their union, execute the no-op mShell_arr_cum[idx+1:] = 0.0 at :190, then any() both full-length arrays at :191-192; has_neutral = is_phiDepleted and not is_allMassSwept at :221 consumes them. Lens C, independently, requires invariant I1: integral(4 pi r^2 rho dr) over [R2, R_out] == M_sh with the residual CHECKED, not assumed (S8-C-09).",
    "expected": "Evaluate the flags at idx only (is_allMassSwept = massCondition[idx], is_phiDepleted = phiCondition[idx]) -- which is presumably what the truncation at :190 was meant to achieve -- and assert |sum(dm) - mShell_end|/mShell_end below a tolerance before returning.",
    "failure_scenario": "phi depletes at index 10 while the unphysical continuation accumulates mShell_end by index 500. is_allMassSwept is set True, has_neutral becomes False, and the neutral shell is never integrated even though the shell is phi-depleted with mass left to sweep. shell_thickness, tau_rEnd, shell_nMax, tau_kappa_IR, rShell and the gravity arrays are then ionised-only values, and the code believes the mass closed.",
    "repro": "Add a pytest case asserting mShell_arr_cum[idx] >= mShell_end whenever the returned ShellProperties reports the mass swept; and assert |collected shell mass - mShell_end|/mShell_end < 1e-6 on several snapshots of param/simple_cluster.param.",
    "confidence": "high",
    "lenses": ["A", "C"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S8-A-03", "S8-A-04", "S8-C-09"]
  },
  {
    "id": "S8-R-04",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 251,
    "class": "deadcode",
    "severity": "S2",
    "claim": "The clamp n_IF_Str = min(n_IF_Str, shell_n0) always binds GIVEN THE RHS AS IMPLEMENTED, so the four-line Stroemgren inversion at :246-249 is always discarded. The proof rests on dn/dr >= 0, which holds only because the implemented RHS omits the inertial and gravity terms (S8-R-05) -- it is not an unconditional property of the physics.",
    "evidence": "Lens A's inequality chain (see S8-R-02 evidence) is algebraically tight: the volume and the photon integral share the same endpoint R_IF = rShell_arr_ion[-1], so grid overshoot cannot break it, and equality occurs only in the exactly-uniform limit. But Lens C's derivation of the same equation (S8.1, invariant I5) states dP/dr >= 0 only 'when f_rad >= 0 AND inertia is neglected'; C's required momentum equation dP/dr = -rho(a_sh + GM/r^2) + f_rad has negative terms that the implemented RHS does not contain.",
    "expected": "Either drop the inversion and return shell_n0 explicitly, or make the cap a logged diagnostic. If the missing -rho a_sh term is ever added (S8-R-05), re-derive: the clamp will no longer always bind.",
    "failure_scenario": "Any consumer treating n_IF_Str as an independent Stroemgren estimate of the front density is comparing against the inner-boundary density constant. Combined with S8-R-02 this is how the P_HII ceiling becomes structural.",
    "repro": "Assert n_IF_Str == shell_n0 exactly on every non-dissolved snapshot of a full run; and log the pre-clamp value to confirm it is always >= shell_n0.",
    "confidence": "high",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S8-A-02", "S8-B-02", "S8-C-01"]
  },
  {
    "id": "S8-R-05",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": 115,
    "class": "coefficient",
    "severity": "S2",
    "claim": "The shell-structure momentum equation contains only radiation terms: dn/dr = (mu_p/(mu_H k_B T_i)) dP_rad/dr. The inertial term -rho a_shell and the gravity term -rho GM(<r)/r^2 required for the structure ODE to be the same equation as the global thin-shell EOM are absent.",
    "evidence": "Lens A transcribes the full RHS: dust-momentum plus recombination-momentum, both non-negative, with no acceleration or gravity term anywhere in get_shellODE.py:94-147. Lens C derives (S8.1) dP/dr = -rho(a_sh + GM/r^2) + f_rad and shows that integrating it across the shell reproduces the thin-shell EOM (SPEC-020) term for term (S8.2, invariant I10) -- 'the single strongest cross-tier invariant in S8'. Lens B has NO prose on the momentum equation at all (S8-B-19).",
    "expected": "Either the inertial/gravity terms are carried, or it is documented that the structure ODE is a radiation-only quasi-static profile whose P(R_out) must NOT be fed back into the dynamics as the physical outer pressure.",
    "failure_scenario": "With the shell accelerating, the true pressure gradient can change sign, moving the density peak from the outer to the inner edge. That inverts shell_nMax's location (feeding the dissolution criterion), the optical-depth weighting, the ionisation-front position -- and removes the premise of S8-R-04's always-binds proof.",
    "repro": "Check closure of 4 pi R2^2 (P_in - P_out) + F_rad - F_grav = M_sh dv2/dt against the recorded snapshot forces; and confirm the stored profile always has dn/dr >= 0 (it will, by construction, which is the point).",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S8-C-01", "S8-A-01"]
  },
  {
    "id": "S8-R-06",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 148,
    "class": "numerical",
    "severity": "S2",
    "claim": "The ionised layer is under-resolved and the front is located by grid scan rather than root-find: the integration domain is sized from the filled-sphere Stroemgren radius R_St, which over-states the shell-annulus ionised thickness by ~3 R2^2/R_St^2, so the number of grid points spanning the front is ~1e4 * R_St^2/(3 R2^2) -- of order a few.",
    "evidence": "Lens A: max_shellRadius = (3 Qi/(4 pi chi_e alpha_B n0^2))^(1/3) + rShell_start at :144, sliceSize = np.min([1, (max_shellRadius - rShell_start)/10]) at :148, rShell_step = sliceSize/1e3 at :149, so the step is ~R_St/1e4; there is no root-find anywhere -- the front is 'located by array scan', phiCondition = phiShell_arr <= 1e-9 at :182, R_IF = rShell_arr_ion[-1] at :226. Lens C independently derives Delta_R_ion = R_St^3/(3 R2^2) (T2/S8-C-13) -- a 3700x over-statement for R_St = 0.14 pc, R2 = 5 pc -- and demands root-finding or two separately integrated segments (T6/S8-C-19). Combining: ~2.6 grid points across the ionised layer for C's example.",
    "expected": "Root-find the front (event-terminated integration or bisection on Delta_R), or use geometric/adaptive spacing sized on the annulus thickness R_St^3/(3 R2^2), not on R_St.",
    "failure_scenario": "R_IF, f_esc, f_dust and n_IF are quantised at the grid spacing and are therefore functions of the hardcoded 1e3/5e3 step counts rather than of physics; a grid-convergence test would show the headline escape fraction drifting with a numerical constant.",
    "repro": "Grid-convergence test: rerun a fixed config with the ionised nsteps doubled and quadrupled in separate processes and compare f_esc, shell_fIonisedDust, R_IF and shell_thickness at matched simulation time -- they must be stable to <<1%. Also log the number of grid points between R2 and R_IF.",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S8-A-16", "S8-C-13", "S8-C-19"]
  },
  {
    "id": "S8-R-07",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 449,
    "class": "state",
    "severity": "S2",
    "claim": "ShellProperties carries no convergence/validity flag, so no consumer can distinguish a converged shell profile from a truncated or garbage one, and no snapshot records a shell-integration status.",
    "evidence": "Lens A enumerates every field assigned at :449-471 and every field of the dataclass -- there is no status field, and both the live and dissolved arms assign only physics values. Lens B transcribes the full documented field list (:46-82) -- three boolean flags (is_shellDissolved, is_phiDepleted, diss_condition_met), none of them a solver status. Lens C requires exactly this (S8-C-21): 'a result object that carries only physics values cannot distinguish converged, f_esc = 0.9 from truncated, f_esc looks like 0.9'.",
    "expected": "A boolean or status enum on ShellProperties, written into the snapshot and counted in metadata, with F_rad, P_drive, the f_esc reporting and the dissolution check all gating on it.",
    "failure_scenario": "A parameter sweep silently mixes converged and failed shells. Because the truncation errors are signed (S8-R-01), the failed subset biases the published grid in one direction rather than adding scatter.",
    "repro": "Inspect ShellProperties fields for any status/validity member; grep consumers for a convergence gate; check whether any snapshot column records a shell-integration outcome.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S8-C-21", "S8-A-01", "S8-B-05"]
  },
  {
    "id": "S8-R-08",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": 94,
    "class": "regime",
    "severity": "S2",
    "claim": "The sharp-front approximation (boolean is_ionised, x == 1 inside / 0 outside) is used with no validity check and no flag, while its validity condition ell_IF/Delta_R_ion = 4 pi R2^2 alpha_B chi_e n_H/(sigma_HI Q_i) reaches ~1.4 at n_H ~ 1e5 cm^-3 -- a density a compressed shell around a cloud with TRINITY's default nCore = 1e5 will reach and exceed.",
    "evidence": "Lens C [computed] with sigma_HI = 6.3e-18 cm^2 [recalled, high] and alpha_B = 2.59e-13 cm^3 s^-1 [recalled, high], for Q_i = 1e49 s^-1 and R2 = 5 pc: ratio 0.14 at n_H = 1e4 and 1.4 at n_H = 1e5. Corroborating absence-of-guard: Lens A covered 100% of both files and reports no ionisation-fraction variable, no x(r), no warning and no flag -- is_ionised is a plain boolean argument; Lens B reports no documented validity range for the x = 1 assumption anywhere in the slice (only 'shell HII is singly ionised').",
    "expected": "Either a continuous x(r) in the stiff regime, or an explicit validity check with a logged warning/flag when the ratio exceeds ~0.3.",
    "failure_scenario": "With x < 1 the true recombination rate is alpha_B chi_e x^2 n^2, i.e. slower, so the true ionised layer is thicker by ~1/x^2. Assuming x = 1 over-consumes the photon budget and under-predicts Delta_R_ion and f_esc. The bias has a fixed sign and grows with density, so it is worst exactly in the dense-cloud runs the paper features.",
    "repro": "First settle reachability: read the nCore default from trinity/_input/ (no lens verified this). Then evaluate 4 pi R2^2 alpha_B chi_e n_H/(sigma_HI Q_i) from stored snapshots of a high-nCore run and report the fraction of snapshots exceeding 0.3.",
    "confidence": "medium",
    "lenses": ["C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S8-C-14"]
  },
  {
    "id": "S8-R-09",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 29,
    "class": "regime",
    "severity": "S2",
    "claim": "The code's own comment names simple_cluster -- the tracked quickstart/baseline config -- as the 'degenerate code-unit-overflow regime' in which the shell integration overflowed and was silently truncated.",
    "evidence": "Lens B, verbatim: shell_structure.py:29-30 identifies simple_cluster; get_shellODE.py:19-31 says the ionisation front peaks at ~1e65 code units and the discarded post-front tail overflows float64, justifying _NSHELL_MAX ~ 1e120. Lens A corroborates structurally: the +n^2 recombination term is a finite-radius pole just past the front, phi has no equilibrium at zero so the state runs to -inf past the front, and the 1e120 clamp exists precisely because the runaway is real.",
    "expected": "The documented single-run example should not be the config that drives the shell solver into a numerical-degeneracy regime; either the code-unit scaling of nShell is revisited or the baseline is relabelled.",
    "failure_scenario": "Every new user's first run, and every regression comparison anchored on simple_cluster, exercises behaviour dominated by overflow guards rather than physics -- and it is the config that demonstrably reached the S8-R-01 failure path at the previous mxstep default.",
    "repro": "Run param/simple_cluster.param with full_output=1 on both odeint calls; record max(nShell) inside get_shellODE (including the discarded tail) against _NSHELL_MAX and float64 max, and record infodict['nst'] against mxstep.",
    "confidence": "high",
    "lenses": ["B", "A"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S8-B-06", "S8-A-07", "S8-A-08"]
  },
  {
    "id": "S8-R-10",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 313,
    "class": "numerical",
    "severity": "S2",
    "claim": "The neutral-region slice size sliceSize = np.min([1, (max_shellRadius - rShell_start)/10]) has no lower bound and rShell_start is R_IF, which by construction approaches max_shellRadius, so sliceSize can be exactly zero (ZeroDivisionError) or negative (inverted integration limits and a non-terminating inward march).",
    "evidence": "Lens A: dphi/dr <= -4 pi r^2 chi_e alpha_B n0^2/Qi with n monotonically increasing, so phi crosses zero at r <= max_shellRadius and the ionised loop stops at the first grid point past that, hence R_IF can equal or slightly exceed max_shellRadius after discretisation (overshoot bounded by one step). np.min([1, x]) does not clamp x from below. VERIFIED by execution: np.arange(5.0, 4.9, -1e-4) yields 1000 descending points; np.arange(5.0, 5.0, 0.0) raises ZeroDivisionError.",
    "expected": "Clamp the slice size to a strictly positive floor, or size the neutral slices from the remaining shell mass / a neutral length scale rather than from the ionised Stroemgren estimate.",
    "failure_scenario": "Low-dust shell whose IF sits at the Stroemgren radius: either (a) ZeroDivisionError from np.arange, or (b) rShell_step < 0 so odeint integrates inward, mShell_arr[1:] at :334 is negative, massCondition is never True and 'while not is_allMassSwept' at :316 loops forever marching toward r = 0 where dn/dr divides by r**2, or (c) tiny-positive sliceSize giving an unbounded number of outer iterations.",
    "repro": "python -c \"import numpy as np; print(np.arange(5.0,4.9,-1e-4)[[0,-1]]); np.arange(5.0,5.0,0.0)\"  ; then instrument :313 to log sliceSize on a low-dust config and assert it is strictly positive.",
    "confidence": "medium",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S8-A-05"]
  },
  {
    "id": "S8-R-11",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 316,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "The neutral integration loop has no radius bound, no optical-depth bound and no iteration cap -- its only exit is mass sweep-up -- so any state with a near-zero neutral density loops forever; the declared tau_max = 100 at :311 is assigned and never read.",
    "evidence": "Lens A: :316 'while not is_allMassSwept:' with is_allMassSwept set only from massCondition = mShell_arr_cum >= mShell_end (:337, :345); tau_max appears exactly once in the file; max_shellRadius is used only for the slice size at :313; the mass increment n*mu_convert*4*pi*r^2*rShell_step at :334 goes to zero with n. Contrast the ionised loop, which is bounded because dphi/dr <= -4 pi r^2 chi_e alpha_B n0^2/Qi guarantees phi crosses zero by max_shellRadius.",
    "expected": "A hard radius / iteration / optical-depth cap with an explicit failure signal (and either implement the tau cutoff or delete tau_max).",
    "failure_scenario": "Reached whenever nShell0 entering the neutral loop is ~0 -- exactly what happens after the silent odeint failure of S8-R-01 (denormals ~1e-310) or after S8-R-10's negative rShell_step makes the mass increments negative. The run hangs with no diagnostic.",
    "repro": "Add an iteration counter to the neutral loop and assert it stays below a generous bound on all tracked configs; the S8-R-01 pytest case with a timeout marker exercises the hang directly.",
    "confidence": "high",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S8-A-20", "S8-A-13"]
  },
  {
    "id": "S8-R-12",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": 111,
    "class": "numerical",
    "severity": "S3",
    "claim": "phi = max(0.0, phi) clamps only the local copy used to build the RHS; the integrator state is untouched and dphi/dr remains strictly negative at phi = 0, so phi has no equilibrium and runs away negative past the front. The clamp makes the RHS piecewise and kinked exactly where the profile is read from, and an arbitrarily large negative overshoot is reported as f_esc_ion = 0 with no diagnostic.",
    "evidence": "Lens A: :111 rebinds the local phi; with phi clamped to 0 the dust sink vanishes but -4 pi r^2 chi_e alpha_B nShell**2/Qi is still strictly negative, and only the caller's array test phiShell_arr <= 1e-9 (:182) bounds the excursion; three further max(0.0, .) clamps at :204 and :229 hide the overshoot. Lens B (contradiction J and S8-B-11): the comments frame the clamp as sign hygiene and claim bit-identical output, and 'neither acknowledges that the ODE being solved is no longer the documented one'; B also records the stray '# <-- add this line' diff instruction at :111 immediately after the clamp comment. Lens C (S8-C-23): positivity/monotonicity guards must be assertions, never clamps.",
    "expected": "Terminate the ionised integration at the front with a terminal event (solve_ivp), or make the RHS self-consistent by zeroing dphi/dr once phi <= 0; report the overshoot magnitude rather than clamping it away.",
    "failure_scenario": "The kink at phi = 0 sits exactly where the phi <= 1e-9 termination test fires, i.e. in the region the profile is read from, degrading LSODA's error control where the system is stiffest -- which raises the probability of the mxstep failure that S8-R-01 consumes silently. C's specific mechanism (clamping the STATE creates a spurious front) is NOT what this code does -- the clamp is on the RHS copy and the front is located by the caller -- but the class of hazard is the same.",
    "repro": "Log min(phiShell_arr) per slice on a stiff config; assert the most negative phi ever reached is within the termination threshold of zero rather than a large negative number.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S8-A-07", "S8-B-11", "S8-B-10", "S8-C-23"]
  },
  {
    "id": "S8-R-13",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": 144,
    "class": "coefficient",
    "severity": "S3",
    "claim": "f_cover is applied as a multiplier on dtau/dr in the ionised branch only (:122) and is absent from the neutral branch (:144); it is hardwired to 1 at the call site, two TODOs admit the fragmentation coupling is missing, and scaling tau by f_cover is not a consistent covering-fraction treatment in the first place.",
    "evidence": "Lens A: :122 'dtaudr = nShell*sigma_dust*f_cover' versus :144 'dtaudr = nShell*sigma_dust'; f_cover is a declared parameter threaded through both odeint calls and hardcoded to 1 at shell_structure.py:115. Lens B: get_shellODE.py:43 documents f_cover with a validity range 0 < f_cover <= 1 while :35 and shell_structure.py:114 both carry TODOs saying it still needs adding. Lens C (S8-C-17): the physically consistent convention leaves the flux per covered area unchanged and ENHANCES the column by 1/f_cover (Sigma_patch = M_sh/(4 pi R2^2 f_cover)), with escape bookkeeping f_esc_total = (1 - f_cover) + f_cover f_esc_patch.",
    "expected": "One covering convention applied once and identically in both branches -- entering the mass/column accumulation and the escape bookkeeping. Note the current form gives exp(-f_cover tau), which is neither the enhanced-column picture nor the correct mixture (1 - f_cover) + f_cover exp(-tau).",
    "failure_scenario": "Inert today (f_cover == 1), but the moment fragmentation is wired in, tau across the neutral region uses a different covering assumption from the ionised region -- so f_absorbed_neu = 1 - exp(-tau_rEnd) (:399) and shell_fAbsorbedWeightedTotal (:400) are inconsistent -- and the correction has the opposite sign to C's derivation (less coverage should thicken the patch column, not thin it).",
    "repro": "Call get_shellODE with f_cover = 1.0 and 0.5 on identical y and r in both branches; the neutral derivatives will be identical (parameter inert) while the ionised ones scale. Then check tau_patch scales as 1/f_cover, not f_cover.",
    "confidence": "medium",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S8-A-06", "S8-B-03", "S8-C-17"]
  },
  {
    "id": "S8-R-14",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 307,
    "class": "regime",
    "severity": "S3",
    "claim": "The ionisation-front density jump is implemented correctly as pressure continuity, n_neu = n_ion (mu_n/mu_p)(T_i/T_n), but the neutral layer is ALWAYS atomic -- mu_atom and TShell_neu with no molecular branch -- so a cold molecular neutral shell gets a jump roughly 1.8x too small.",
    "evidence": "Lens A: the neutral loop is seeded at :307-308 with nShell0 scaled by (mu_n/mu_p)(T_i/T_n), and the ODE's own pressure/density conversion is n = mu_p P/(mu_H k_B T) (:124-125), so the seeding IS pressure continuity; only mu_atom appears, never a molecular mean molecular weight. Lens C derives the same jump independently (S8.8: n_neu/n_ion = psi_ion T_ion/(psi_neu T_neu), with mu = mu_H/psi so mu_n/mu_p == psi_ion/psi_neu) and gives 200 for atomic 100 K versus 3667 for molecular 10 K [computed]; T7/S8-C-20 warns that choosing atomic where the gas is molecular is a further factor ~1.8.",
    "expected": "The neutral composition (psi_atom = 1.1 vs psi_mol = 0.6, and the matching T_neu) chosen to match the swept gas state, or an explicit statement that the swept neutral shell is treated as atomic.",
    "failure_scenario": "A GMC shell whose swept gas is molecular at ~10 K gets a neutral density ~1.8x too low, hence shell_nMax under-predicted (biasing the dissolution trigger toward firing), and tau_UV/tau_IR under-predicted (radiation force under-predicted).",
    "repro": "Read a stored shell profile and check psi*n*T continuity across the front, then compare n_neu/n_ion against 200 (atomic, 100 K) and against the value implied by the configured TShell_neu and mu_atom.",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S8-C-06", "S8-C-20", "S8-A-01"]
  },
  {
    "id": "S8-R-15",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 446,
    "class": "state",
    "severity": "S3",
    "claim": "The dissolution state machine is inconsistent: the dissolved arm sets nShell_max exactly equal to nISM (:423) while diss_condition_met tests the strict inequality nShell_max < nISM (:446), so a dissolved shell always reports False; meanwhile the 'dissolved' gate acting at :417/:428/:430 must be an input from a prior timestep, since the fresh condition is only evaluated at :441.",
    "evidence": "Lens A: :423 'nShell_max = params[nISM].value'; :446 'diss_condition_met = bool(allow_dissolution and nShell_max < nISM)'; nISM < nISM is False. Lens B: :209-210 'Dissolution condition is now evaluated AFTER shell structure is computed; shell_structure_pure is stateless', while the dissolved branch is consumed at :417/:428/:430 -- a one-step lag no comment names -- and two separate flags exist for the state ('Is the shell dissolved?' :69 and 'Is shell_nMax < nISM this timestep?' :71).",
    "expected": "Decide whether the flag means 'the shell just dissolved' or 'the shell is dissolved' and make the dissolved arm consistent with that; document the one-step lag explicitly if it is intended.",
    "failure_scenario": "If a downstream latch re-reads diss_condition_met each step to decide whether the shell REMAINS dissolved, it reads False on every step after the first and may un-dissolve the shell; combined with the lag, the shell can also be treated as an absorber for one extra timestep after dissolving (f_esc under-reported, radiation force over-reported).",
    "repro": "Trace which flag gates :417/:428/:430 and whether it originates from params (previous step) or from :441; then check a run that dissolves and confirm diss_condition_met on the steps after dissolution.",
    "confidence": "medium",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S8-A-19", "S8-B-17"]
  },
  {
    "id": "S8-R-16",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 157,
    "class": "deadcode",
    "severity": "S3",
    "claim": "The full ionised ODE integration loop runs unconditionally even when the shell is already flagged dissolved, and all of its results are then overwritten by constants.",
    "evidence": "Lens A: is_shellDissolved is read at :130-132 but the while loop at :157-218 and the derived quantities at :224-253 execute regardless; the dissolved arm at :416-437 overwrites n_IF, n_IF_ODE, R_IF and n_IF_Str with 0.0 and sets shell_r_arr/shell_n_arr to empty arrays.",
    "expected": "Take the dissolved branch before integrating.",
    "failure_scenario": "A dissolved shell pays the full integration cost and remains exposed to the failure modes of that integration (S8-R-01 hang, S8-R-17 crash) on a code path whose output is discarded -- so a crash or hang can occur in exactly the regime the code intends to short-circuit.",
    "repro": "Time shell_structure_pure on a dissolved-state params dict and confirm the ionised loop runs; assert the returned n_IF is 0.0 regardless.",
    "confidence": "high",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S8-A-10"]
  },
  {
    "id": "S8-R-17",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 148,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "Qi = 0 makes max_shellRadius == rShell_start, hence sliceSize = 0 and rShell_step = 0, so np.arange raises ZeroDivisionError before the ODE is entered; Qi = 0 would also divide by zero at get_shellODE.py:117 and :120.",
    "evidence": "Lens A: :144 max_shellRadius = (3*Qi/(4*pi*chi_e*alpha_B*nShell0**2))**(1/3) + rShell_start; :148 sliceSize = np.min([1, 0/10]) = 0; :149 rShell_step = 0; :161 np.arange(a, a, 0) raises (verified). get_shellODE.py:117 has Li/Qi and :120 divides by Qi with no guard.",
    "expected": "Guard the no-ionising-photon regime explicitly -- it is a physically reachable late-time state -- rather than relying on an arange exception.",
    "failure_scenario": "A cluster whose ionising output has switched off (post-SN / late evolution) enters shell_structure_pure and crashes with ZeroDivisionError from np.arange instead of taking the dissolved/neutral path.",
    "repro": "python -c \"import numpy as np; np.arange(5.0,5.0,0.0)\"  ; then call shell_structure_pure with Qi = 0 in a realistic params dict.",
    "confidence": "medium",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S8-A-09"]
  },
  {
    "id": "S8-R-18",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": 3,
    "class": "citation",
    "severity": "S3",
    "claim": "The module's three governing equations are never written in prose and have no literature citation; only three isolated terms appear, incidentally, inside numerical-guard comments.",
    "evidence": "Lens B: get_shellODE.py:3, :19 ('+nShell**2'), :109 ('-n*sigma_d*phi'), :110 ('Li*phi'); no pressure-balance equation, no coefficient, no temperature dependence, no reference anywhere in the slice. Lens A reconstructs from code a substantive system whose dn/dr carries the specific prefactor mu_p/(mu_H k_B T_i) -- a modelling choice ('isothermal shell in radiative-pressure hydrostatic balance') that no comment states and no citation supports.",
    "expected": "The three RHS expressions written out with their coefficients, plus a citation for the shell momentum/pressure-balance equation.",
    "failure_scenario": "The core equation of the module is unauditable from prose: every coefficient, temperature, mean molecular weight and sigma_d in dn/dr is unchecked, and a sign or factor error there cannot be caught by review. Only the '+' of the n^2 term is documented, and only because it happens to cause an overflow.",
    "repro": "",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S8-B-19"]
  },
  {
    "id": "S8-R-19",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 232,
    "class": "citation",
    "severity": "S3",
    "claim": "The only literature citation in the slice is 'Lancaster+2025' (:77 and :232), with no equation number, journal, arXiv id or page, and qualified as 'generalised' with the nature of the generalisation unstated.",
    "evidence": "Lens B: shell_structure.py:77 'Stroemgren ionization balance density (Lancaster+2025), sole source of P_HII'; :232 'Lancaster+2025, generalised'. Attributed to it: the formula, the two-regime dV/f_esc substitution table, and by adjacency the n_IF_Str <= shell_n0 cap. Lens C derives the underlying balance independently and gets the annulus form the code uses, but with literature access blocked could verify nothing.",
    "expected": "A resolvable reference plus the specific equation number, and an explicit statement of what 'generalised' changed relative to the source -- specifically whether the (1 - f_esc) factor, the R^3 - R2^3 shell form and above all the shell_n0 cap come from the paper or are local inventions.",
    "failure_scenario": "The formula cannot be checked against its source, which is exactly the ambiguity that leaves S8-R-02 (the cap that makes P_HII <= P_b by construction) unresolvable from the repository alone.",
    "repro": "",
    "confidence": "high",
    "lenses": ["B"],
    "divergence": "BC",
    "status": "single-lens",
    "source_ids": ["S8-B-14"]
  },
  {
    "id": "S8-R-20",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 61,
    "class": "units",
    "severity": "S3",
    "claim": "The ShellProperties field documented as 'tau_IR / kappa_IR = integral(rho dr)' stores a mass column in code units (Msun/pc^2) with no unit annotation, unlike the neighbouring shell_r_arr '[pc]' and shell_n_arr '[1/pc^3]'.",
    "evidence": "Lens A's dimensional table: tau_kappa_IR = mu_H * sum(n * dr) = Msun pc^-2, a column density. Lens B: shell_structure.py:61 gives no units for kappa_IR or rho at the site, and none for the field. Lens C: kappa_IR is conventionally quoted in cm^2 g^-1 while the module is in [Msun, pc, Myr]; sigma_d (per H) and kappa_IR (per gram) differ by mu_H m_H = 2.34e-24.",
    "expected": "An explicit unit on the stored column and on the kappa_IR the consumer must multiply by.",
    "failure_scenario": "A consumer multiplying a code-unit column (Msun/pc^2) by a cgs opacity (cm^2/g) without conversion is wrong by roughly four orders of magnitude, silently switching the IR-trapping term between negligible and dominant.",
    "repro": "Find where this field is multiplied by kappa_IR and check the unit of each factor against trinity/_functions/unit_conversions.py; anchor: Sigma = 0.25 g cm^-2 must give tau_IR = 1 for kappa_IR = 4 cm^2 g^-1.",
    "confidence": "medium",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S8-B-12", "S8-C-11"]
  },
  {
    "id": "S8-R-21",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": 115,
    "class": "units",
    "severity": "S3",
    "claim": "The dimensional structure of the RHS is verified consistent, but the NUMERIC values of the cgs micro-physics constants after conversion to code units (dust_sigma, caseB_alpha, chi_e_shell, mu_convert, mu_ion_shell) are not visible anywhere in this slice and have never been anchor-tested.",
    "evidence": "Lens A checked every term dimensionally against unit_conversions.py and found NO imbalance (dn/dr = pc^-4, dphi/dr = dtau/dr = pc^-1, R_max = pc, mShell = Msun, grav_phi = pc^2 Myr^-2). Lens C warns this module is 'the single largest cgs/AU boundary in the code' and that a missing pc->cm factor in tau = sigma_d n dr fails SILENTLY (tau ~ 1e-18, f_esc -> 1, no radiation force) rather than loudly. A's check constrains the exponents, not the magnitudes; the constants themselves live in params, outside the slice.",
    "expected": "Anchor unit tests pinning the converted constants: alpha_B = 2.59e-13 cm^3 s^-1 = 2.78e-55 pc^3 Myr^-1 [recalled/computed]; sigma_d = 1.5e-21 cm^2 = 1.58e-58 pc^2; 1 cm^-3 = 2.938e55 pc^-3; and an end-to-end anchor tau_UV = 4.63 for n_H = 1e3 cm^-3 over 1 pc.",
    "failure_scenario": "A wrong conversion factor produces finite, plausible-looking output (f_abs ~ 0, f_esc ~ 1, or a front misplaced by orders of magnitude) that no dimensional check can catch.",
    "repro": "Feed get_shellODE a hand-computed state with a known cgs answer: Delta_R_ion = Q/(4 pi R2^2 alpha_B chi_e n^2) = 1.17e14 cm for Q = 1e49 s^-1, R2 = 5 pc, n = 1e4 cm^-3; compare against the code's front location in the dust-free limit.",
    "confidence": "medium",
    "lenses": ["C"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S8-C-11", "S8-C-12"]
  },
  {
    "id": "S8-R-22",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 57,
    "class": "divergence",
    "severity": "S3",
    "claim": "shell_fIonisedDust is computed as phi_dust/(phi_dust + phi_H), i.e. a fraction of the ABSORBED ionising photons, but is documented as 'Fraction of ionizing radiation absorbed by dust', which reads as a fraction of the total ionising luminosity.",
    "evidence": "Lens A: :276-288 re-integrate the two sink terms of dphi/dr with a left-endpoint rectangle rule and return the ratio f_ionised_dust = phi_dust/(phi_dust + phi_H), guarded against a zero denominator. Lens B: :54 'Fraction of ionizing radiation absorbed' vs :57 'Fraction of ionizing radiation absorbed by dust', with no statement of whether the latter is normalised to the total or to the already-absorbed part, and no weighting formula for the luminosity-weighted total at :56.",
    "expected": "The docstring should state the normalisation explicitly (fraction of absorbed, not of total) and give the weighting formula for shell_fAbsorbedWeightedTotal.",
    "failure_scenario": "A consumer splitting the ionising budget into dust and hydrogen channels with the wrong normalisation is off by exactly f_abs,ion, feeding the radiation force and the IR-trapping term. Separately, the rectangle-rule re-integration is first-order and independent of the LSODA-integrated phi, so f_gas + f_dust + f_esc will not close to machine precision (Lens C invariant I2) even though the ODE form is correct.",
    "repro": "Assert f_ion_dust <= f_ion_absorbed <= 1 and check whether phi_dust + phi_H reproduces (1 - f_esc_ion) to better than 1e-6 on stored snapshots.",
    "confidence": "medium",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S8-B-23", "S8-C-05"]
  },
  {
    "id": "S8-R-23",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 82,
    "class": "state",
    "severity": "S3",
    "claim": "shell_ion_idx uses -1 as the 'empty ionised region' sentinel, which in Python is a valid index (the last element).",
    "evidence": "Lens B: :82 'Last index of ionized region in shell_r/n_arr (-1 if empty)'; :405-407 reuse the field for the fully-ionised test 'if shell_ion_idx == len(shell_r_arr)-1, the entire shell is ionized'. Lens A confirms the literals 1 and -1 are assigned at :408 and :437.",
    "expected": "A sentinel that cannot be confused with a valid index, or every consumer explicitly testing shell_ion_idx < 0 before indexing.",
    "failure_scenario": "A consumer doing shell_n_arr[shell_ion_idx] on an empty ionised region silently reads the outermost neutral cell, so 'no ionised region' is reported as 'ionisation front at the shell outer edge with the neutral outer density'. arr[:shell_ion_idx] with -1 silently drops the last element rather than yielding nothing.",
    "repro": "Construct a timestep with no ionised region and grep every consumer of shell_ion_idx for a guarded (< 0) read before indexing or slicing.",
    "confidence": "medium",
    "lenses": ["B"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S8-B-08"]
  },
  {
    "id": "S8-R-24",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 392,
    "class": "regime",
    "severity": "S3",
    "claim": "Nothing checks or flags the thin-shell validity condition Delta_R/R2 << 1, even though shell_thickness is computed and returned.",
    "evidence": "Lens C (S8-C-22, invariant I8): in the normal regime Delta_R/R2 ~ 1e-5 to 1e-3, but as the shell decompresses toward dissolution Delta_R grows without bound while R2 is fixed, and there is no internal mechanism stopping Delta_R from exceeding R2; a shell with Delta_R > R2 breaks the 4 pi R2^2 area factors and any (R_out^3 - R2^3) approximation. Lens A covered 100% of the slice and reports no such check. Mitigating: Lens A's transcription shows the ODE itself uses the local 4 pi r^2 throughout, not 4 pi R2^2, so the STRUCTURE module is fully spherical -- the assumption bites in the EOM and the P_HII geometry, outside this slice.",
    "expected": "An explicit thin-shell validity check (e.g. Delta_R/R2 < 0.1) that flags rather than silently continues.",
    "failure_scenario": "The late-time, thick, decompressing shell -- exactly the regime that decides dispersal vs re-collapse and triggers the dissolution stop -- is where the thin-shell assumptions used downstream fail, with nothing recording it.",
    "repro": "Plot shell_thickness/R2 over a full run of param/simple_cluster.param and of the hidens edge config; report the maximum and whether anything reacts to it.",
    "confidence": "medium",
    "lenses": ["C"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S8-C-22"]
  },
  {
    "id": "S8-R-25",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 3,
    "class": "state",
    "severity": "S3",
    "claim": "The purity/statelessness contract -- 'returns a dataclass instead of mutating the params dictionary', 'read-only access', 'No dictionary mutations during calculation', 'shell_structure_pure is stateless' -- is asserted five times in prose and has no regression test.",
    "evidence": "Lens B: shell_structure.py:3-16, :40-45, :86-101, :102, :210. It is the module's stated reason for existing ('essential for use with adaptive ODE solvers'). Lens A reports no writes to params anywhere in the slice, which supports the contract holding today but is not a guard against regression.",
    "expected": "A pytest case asserting the contract, since the outer adaptive solver's correctness depends on it.",
    "failure_scenario": "An adaptive solver evaluates the shell structure at trial steps that are later rejected. If any write to params or to a module-level global leaks -- project CLAUDE.md explicitly warns 'trinity leaks module-level global state in-process' -- a rejected trial step permanently contaminates the accepted trajectory: a history-dependent, non-reproducible error no single-step equivalence test would catch.",
    "repro": "Deep-copy params, call shell_structure_pure, assert the copy compares equal to the original; call it twice with identical inputs and assert bit-identical ShellProperties on the second call.",
    "confidence": "medium",
    "lenses": ["B"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S8-B-24"]
  },
  {
    "id": "S8-R-26",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": 100,
    "class": "numerical",
    "severity": "S4",
    "claim": "nShell = min(nShell, 1e120) saturates the density used to build the RHS while the integrator's own state keeps growing, so the returned derivative is not the derivative of anything the solver is tracking; the clamp exists in the ionised branch and not the neutral one, and the 'bit-identical' justification is sourced only to an unverified docs/dev writeup.",
    "evidence": "Lens A: :32 defines _NSHELL_MAX = 1e120, :100 applies it after unpacking y at :96; the neutral unpack at :131 has no analogue; nShell**2 = 1e240 times the prefactor can still overflow. Lens B: :26-30 'a NUMERICAL safety rail, NOT a physics cutoff' and 'the consumed shell profile is bit-identical to the unguarded solve (verified end-to-end, docs/dev/shell-solver/OVERFLOW_FIX_PLAN.md)'; :31 claims cap^2 * 1e55 = 1e295 is 'well under' 1.8e308 -- ~13 decades of headroom on a quantity the same comment says spans 55 decades.",
    "expected": "Either clamp the state via a bounded solver/event, or drop the clamp and let the failure surface; and re-derive the bit-identical claim independently of docs/dev (project CLAUDE.md declares that tree unverified).",
    "failure_scenario": "In a runaway the integrator sees a derivative consistent with n = 1e120 while its own state exceeds it, so step-size control is driven by a fiction -- feeding the mxstep failure that S8-R-01 consumes silently.",
    "repro": "Read the literal _NSHELL_MAX and assert _NSHELL_MAX**2 * (max observed dndr prefactor) < 1e300 across the edge configs; re-run a config with and without the guard in separate processes and diff the consumed profile and dictionary.jsonl byte-for-byte at matched simulation time.",
    "confidence": "medium",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S8-A-08", "S8-B-11", "S8-B-21"]
  },
  {
    "id": "S8-R-27",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": 103,
    "class": "numerical",
    "severity": "S4",
    "claim": "The tau > 500 guard replaces exp(-tau) with exactly 0, introducing a step discontinuity in the ODE right-hand side at tau = 500 even though exp(-500) = 7.1e-218 is perfectly representable; underflow only begins near tau = 745.",
    "evidence": "Lens A, VERIFIED: np.exp(-500.0) = 7.124576406741286e-218 and np.exp(-745.0) = 5e-324, both finite. The guard is duplicated at :103-106 (ionised) and :134-137 (neutral) with the same threshold. Lens B (S8-B-13) flagged that the comment 'prevent underflow for very large tau values' names no threshold and worried the two branches might differ -- Lens A refutes that: both use 500.",
    "expected": "Raise the threshold to ~700 where exp genuinely underflows, or drop the branch (np.exp underflows to 0.0 silently anyway).",
    "failure_scenario": "Negligible in magnitude (O(1e-218) jump in dn/dr) but it is a non-smooth RHS crossing an adaptive stiff solver can waste steps on.",
    "repro": "python -c \"import numpy as np; print(np.exp(-500.0), np.exp(-745.0))\"",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S8-A-17", "S8-B-13"]
  },
  {
    "id": "S8-R-28",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 190,
    "class": "deadcode",
    "severity": "S4",
    "claim": "mShell_arr_cum[idx + 1:] = 0.0 has no observable effect.",
    "evidence": "Lens A: massCondition (:181) is an independent boolean array captured before the mutation; every subsequent read is mShell_arr_cum[:idx] (:196) or mShell_arr_cum[idx] (:206, :214); the array is freshly produced by np.cumsum at :178 so no other reference is live; the neutral loop has no analogous statement.",
    "expected": "Either remove it, or move it before :191 and re-derive the flags from the truncated array -- which is the fix for S8-R-03.",
    "failure_scenario": "",
    "repro": "",
    "confidence": "high",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S8-A-04"]
  },
  {
    "id": "S8-R-29",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 225,
    "class": "deadcode",
    "severity": "S4",
    "claim": "n_IF_ODE is an unconditional duplicate of n_IF on every code path; both are returned as separate ShellProperties fields.",
    "evidence": "Lens A: :224-225 'n_IF = nShell_arr_ion[-1]; n_IF_ODE = n_IF'; the dissolved arm sets both to 0.0 at :431-432; both are passed through at :465-466. Lens B: :75 documents the field as 'Same as n_IF (raw ODE value, kept for diagnostics)' and worried the two could diverge on the dissolved path -- Lens A refutes that: both are zeroed identically.",
    "expected": "One field, or two genuinely different estimates.",
    "failure_scenario": "",
    "repro": "Assert the two fields are equal on every returned ShellProperties across a full run, including dissolved and phi-depleted timesteps.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S8-A-14", "S8-B-07"]
  },
  {
    "id": "S8-R-30",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 175,
    "class": "deadcode",
    "severity": "S4",
    "claim": "mShell_arr_ion, mShell_arr_cum_ion, mShell_arr_neu and mShell_arr_cum_neu are accumulated across both loops and never read; and element 0 of mShell_arr holds a CUMULATIVE mass while elements 1: hold DIFFERENTIAL masses, so those arrays mix the two quantities at every slice boundary.",
    "evidence": "Lens A: mShell_arr_ion appears only at :136, :195, :213; mShell_arr_cum_ion at :137, :196, :214; the neutral pair at :291/:347/:359 and :292/:348/:360; none appear in the ShellProperties construction at :449-471. Separately :175 'mShell_arr[0] = mShell0' (a running total from the previous slice) versus :176-177 'mShell_arr[1:] = n*mu*4*pi*r^2*rShell_step' (per-cell). The cumsum at :178 is correct only because of this construction.",
    "expected": "Remove the unused arrays, or keep the running offset out of the per-cell array (mShell_arr_cum = mShell0 + np.cumsum(dm)) so the collected array has a single meaning.",
    "failure_scenario": "Latent while the arrays are unused; it becomes a real error the moment anyone reads mShell_arr_ion as a mass profile -- e.g. to check Lens C's mass-closure invariant I1.",
    "repro": "",
    "confidence": "high",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S8-A-11", "S8-A-12"]
  },
  {
    "id": "S8-R-31",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 413,
    "class": "state",
    "severity": "S4",
    "claim": "In the no-neutral case three returned fields alias the same ndarray buffer: shell_r_arr, shell_grav_r and the internal rShell_arr_ion are the same object (likewise shell_n_arr and nShell_arr_ion).",
    "evidence": "Lens A: :263 grav_ion_r = rShell_arr_ion; :273 grav_r = grav_ion_r; :413-414 shell_r_arr = rShell_arr_ion, shell_n_arr = nShell_arr_ion -- none of these copy. Only the has_neutral path (:376-377, :410-411) produces fresh arrays via np.concatenate.",
    "expected": "Copy on return, or document the aliasing.",
    "failure_scenario": "Any downstream in-place edit (a unit conversion applied with *=, sorting, clipping) to one field silently mutates the other, and only in the no-neutral case -- so the bug would be regime-dependent and hard to reproduce.",
    "repro": "assert props.shell_r_arr is not props.shell_grav_r on a no-neutral snapshot.",
    "confidence": "high",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S8-A-15"]
  },
  {
    "id": "S8-R-32",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 148,
    "class": "units",
    "severity": "S4",
    "claim": "The slice-size cap is a bare dimensioned literal: np.min([1, ...]) at :148 and :313 caps the radial slice at 1 pc with nothing in the arithmetic marking it as a length.",
    "evidence": "Lens A: the code's length unit is pc (unit_conversions.py cm2pc, v_kms2au = 1.0227 pc/Myr), so the 1 is 1 pc while the other argument is a genuine radius difference.",
    "expected": "A named constant with its unit stated, or a cap derived from a physical length scale.",
    "failure_scenario": "The literal silently sets the resolution of the whole shell profile and of the mass/gravity quadratures at :176, :264, :277-282, :389-395 -- see S8-R-06.",
    "repro": "",
    "confidence": "high",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S8-A-16"]
  },
  {
    "id": "S8-R-33",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 400,
    "class": "silent-failure",
    "severity": "S4",
    "claim": "The luminosity-weighted absorbed fraction divides by (Li + Ln) with no zero guard, on both the live path (:400) and the dissolved path (:419).",
    "evidence": "Lens A: identical expressions at :400 and :419; on the dissolved path the numerator is identically 0, so the result is 0/0.",
    "expected": "Guard Li + Ln == 0 and return 0 (or NaN with a flag).",
    "failure_scenario": "A cluster with no remaining radiative output raises ZeroDivisionError -- and does so even on the dissolved path, which is precisely the state such a cluster would be in.",
    "repro": "",
    "confidence": "medium",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S8-A-18"]
  },
  {
    "id": "S8-R-34",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": 43,
    "class": "other",
    "severity": "S4",
    "claim": "Docstring and debris hygiene: only the 3-component ionised state vector is documented although the neutral branch takes y = [nShell, tau] and returns a 2-tuple; the independent variable r is documented as 'list -- an array of radii' although an odeint RHS receives a scalar; a stray '# <-- add this line' diff instruction is committed at :111; 'paramters' typo at :128.",
    "evidence": "Lens B: :3 and :43 give the 3-component order and qualify dphidr as '(only in ionised region)'; the neutral branch comments run unravel (:130) -> number density (:139) -> optical depth (:143) -> return (:146) with no phi step. Lens A confirms both shapes exist and that BOTH call sites pack y correctly today (:165-171 with [n, phi, tau]; :324-329 with [n, tau]) -- so this is documentation drift, not a live defect.",
    "expected": "Document both shapes and arities explicitly; r documented as a float; the diff marker and typo removed.",
    "failure_scenario": "A future edit packs the neutral IC as [n, phi, tau] or unpacks a 2-tuple as three values. The silent variant: neutral y unpacked as (nShell, tau) while the caller passed (nShell, phi), corrupting the exp(-tau) attenuation with no exception.",
    "repro": "",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S8-B-04", "S8-B-18", "S8-B-10"]
  },
  {
    "id": "S8-R-35",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 306,
    "class": "other",
    "severity": "S4",
    "claim": "The ionisation-front jump applied at :307-308 is pressure continuity and is physically correct, but the comment records only 'Temperature/density discontinuity at boundary' with no factor, no temperatures and no justification.",
    "evidence": "Lens A gives the applied factor (mu_n/mu_p)(T_i/T_n); Lens C derives independently that this is exactly the weak-D-front pressure-continuity condition n_neu/n_ion = psi_ion T_ion/(psi_neu T_neu) (with mu = mu_H/psi, the two forms are identical) and computes 200 for atomic 100 K; Lens B records the bare comment at :306. All three agree on the physics; only the documentation is missing.",
    "expected": "The jump condition written out with the temperatures named and their source in params identified, plus the weak-D-front justification.",
    "failure_scenario": "An unstated factor is unauditable: a future edit dropping mu_n/mu_p (keeping only T_i/T_n) halves the neutral density with nothing to catch it -- Lens C's trap T7.",
    "repro": "Extract the multiplicative factor applied at the boundary and check psi*n*T continuity across the front on a stored profile.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S8-B-16", "S8-C-06"]
  },
  {
    "id": "S8-R-36",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 428,
    "class": "other",
    "severity": "S4",
    "claim": "'Keep previous rShell value when dissolved (matches original behavior)' (:428), set up by 'Capture previous rShell for dissolved case (original doesn't update rShell when dissolved)' (:111) -- deliberate bug-compatibility with an unnamed predecessor, no rationale.",
    "evidence": "Lens B: shell_structure.py:111 and :428.",
    "expected": "Either a physical justification for freezing rShell at dissolution, or an explicit acknowledgement that this is bug-compatibility being preserved deliberately, with 'the original' identified.",
    "failure_scenario": "Downstream consumers reading rShell after dissolution get a stale radius, with no way to distinguish a frozen from a live value except the dissolved flag.",
    "repro": "",
    "confidence": "high",
    "lenses": ["B"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S8-B-20"]
  },
  {
    "id": "S8-R-37",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": 115,
    "class": "citation",
    "severity": "S4",
    "claim": "Neglect of resonantly trapped Lyman-alpha radiation pressure is a defensible simplification but is silently absent rather than documented as a known neglected term.",
    "evidence": "Lens C: case B implies ~0.68 Lyman-alpha photons per recombination [recalled, medium]; resonant scattering can multiply their momentum deposition substantially at high column. Lens A's transcription of dn/dr shows only the dust-momentum and recombination-momentum terms, confirming the absence. Dust absorption of Lyman-alpha is what normally prevents runaway, so the omission is usually defensible.",
    "expected": "A documented statement of the neglected term and its regime of validity (high dust-to-gas, moderate column).",
    "failure_scenario": "In dust-poor or very high-column regimes the omitted Lyman-alpha force is a real under-estimate of F_rad, and nothing tells a user the model does not apply there.",
    "repro": "",
    "confidence": "low",
    "lenses": ["C"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S8-C-24"]
  },
  {
    "id": "S8-R-38",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 395,
    "class": "other",
    "severity": "S4",
    "claim": "Hand-off to the force-budget slice: three Lens C requirements concern consumers of this slice's outputs, not the slice itself -- (a) reprocessed IR must multiply the ABSORBED luminosity, (L/c)(1-e^-tau_UV)(1+tau_IR), not the additive form; (b) ionising-band momentum must be counted once, not both via <hnu>Q_i/c and inside L_bol/c; (c) sigma_d and kappa_IR must carry the same metallicity scaling.",
    "evidence": "Lens C S8-C-15, S8-C-16, S8-C-25. Lens A confirms these are out of scope for S8: the slice returns only the mass column tau_kappa_IR = integral(rho dr) and the two band-resolved absorbed fractions, and within the ODE both bands are attenuated separately (Ln*exp(-tau) and Li*phi) with no double count; the IR force and the band bookkeeping are computed elsewhere.",
    "expected": "Carry these three checks into the force-budget/radiation slice rather than closing them here.",
    "failure_scenario": "Deferred -- see the C items. Magnitudes per C: the IR form differs by <1% in the UV-thick regime but diverges in thin/dissolving shells; LyC double-counting is a 10-30% force error.",
    "repro": "",
    "confidence": "medium",
    "lenses": ["C"],
    "divergence": "scope-creep",
    "status": "single-lens",
    "source_ids": ["S8-C-15", "S8-C-16", "S8-C-25"]
  },
  {
    "id": "S8-R-39",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 182,
    "class": "other",
    "severity": "S4",
    "claim": "The ionised-region termination threshold is described only as a 'small positive threshold' in the file that applies it, while the specific value 1e-9 is documented in the other file.",
    "evidence": "Lens B: shell_structure.py:182 'small positive threshold'; get_shellODE.py:22 'shell_structure truncates the profile AT the front (first phi<=1e-9 / mass-limited row)'. Lens A confirms the literal 1e-9 lives at shell_structure.py:182.",
    "expected": "The numeric threshold documented (or named as a shared constant) where it is used, with the get_shellODE comment referring to it.",
    "failure_scenario": "The threshold is changed in shell_structure.py and the entire justification for _NSHELL_MAX in get_shellODE.py:19-31 -- which depends on how far past the front the discarded tail extends -- goes stale.",
    "repro": "",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S8-B-15"]
  }
]
```
