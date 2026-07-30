# S2 cloud properties — reconciled

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

**Status (2026-07-30):** 📗 reconciled report — merges `S2_cloud_lensA.md` (what the code does),
`S2_cloud_lensB.md` (what it claims), `S2_cloud_lensC.md` (what it should be). The reconciler read
**only** those three files; no source was consulted. Every claim below inherits its lens's
verification status, which is stated per item.

---

## 0. Reading the evidence base

- **Lens A** executed re-implementations of the coded formulas standalone (no trinity import) and
  quotes measured numbers. Its factual transcriptions are the strongest evidence in the set.
- **Lens B** saw only prose. It can establish what is *claimed*, never what runs.
- **Lens C** saw signatures plus a physics spec and derived everything, cross-checking against its
  own quadrature and its own Lane–Emden integration. Its power-law algebra is derivable from
  scratch, so an A≠C divergence there is strong evidence. Its Bonnor–Ebert constants *are* shown as
  a computation (DOP853, `rtol=1e-12`, Brent maximisation of `m_P`), and **Lens A independently
  re-solved the same ODE and matched them** — so those numbers are treated as corroborated by
  computation, not recalled. What remains unconfirmed for BE is only the *naming* convention
  (which literature symbol each constant carries), since neither lens had literature access.

---

## 1. Coverage table

| Topic | Lens A (does) | Lens B (claims) | Lens C (should) | Triangulated? |
|---|---|---|---|---|
| PL density ρ(r), core + envelope | ✓ transcribed, incl. tanh blend | ✓ (sharp step form) | ✓ derived (sharp step form) | yes |
| PL enclosed mass, all branches | ✓ transcribed **+ own integration** | ✓ (4 identical statements) | ✓ derived **+ quadrature** | **yes — 3-way** |
| α = −3 singular branch | ✓ executed → NaN; derived log form | ✓ (absent here, ValueError there) | ✓ derived log form | **yes — 3-way** |
| Core/envelope join at rCore | ✓ continuous by construction | ✓ same | ✓ same | yes |
| Cloud/ISM join at rCloud | ✓ tanh ramp, width 0.01·rCloud | ✓ contradictory prose (step *and* bridge) | ✗ spec has a sharp step only | partial (A+B) |
| Exterior accumulation r > rCloud | ✓ accumulating | ✓ accumulating | ✓ accumulating (called its top check) | **yes — 3-way, passes** |
| Ṁ = 4πr²ρṙ | ✓ measured against dM/dr | ✓ claimed EXACT for all profiles | ✓ derived, local ρ | yes |
| rCloud inversions (both branches) | ✓ verified exact algebraic inverses | ✓ formulas | ✓ closed forms + K(f,α) | yes |
| BE Lane–Emden ODE + series ICs | ✓ re-solved | ✓ | ✓ derived | **yes — 3-way** |
| BE critical constants | ✓ re-computed | ✓ (labels contradict) | ✓ computed | yes |
| BE enclosed mass (analytic) | ✓ verified by quadrature | ✓ | ✓ derived | yes |
| BE numerical fallback | ✓ executed → broken | ✓ (0.5 % claim; "must be sorted") | ✓ (dropped 0→r[0] contribution) | **yes — 3-way** |
| Validators / constraint logic | ✓ executed all three tests | ✓ documented policy | ✓ prescribed order | yes |
| Units / μ / ρ↔n | ✓ full dimensional audit | ✓ contradictions catalogued | ✓ constants derived | yes |
| `initial_profile` reconstruction | ~ partial (mock-dict keys) | ✓ claims catalogued | ~ grid + nEdge semantics | weak |
| Suggestion helpers | ✓ read | ✓ | ~ one invariant | weak |

Gaps no lens covered: the `.param` **schema range** for `densPL_alpha` (C asserts −2 ≤ α ≤ 0 from
the spec; A and B could not see the schema — it is out of slice). This single unverified fact sets
the severity of two findings below.

---

## 2. Mass–density consistency — the primary check

**Question:** is the coded `M(r)` the exact integral `∫₀ʳ4πr′²ρ(r′)dr′` of the coded `ρ(r)`, branch
by branch? Three independent accounts: A transcribed and integrated the code; B recorded the
docstring formulas; C derived the mathematics from scratch.

### 2.1 Branch-by-branch verdict

| Branch | Coded (A) | Documented (B) | Derived (C) | Verdict |
|---|---|---|---|---|
| **Uniform core**, r ≤ rCore (α ≠ 0) | `(4/3)πr³ρ_core` | identical | `(4π/3)ρ_core r³` | **CONSISTENT** — exact integral of a constant ρ. 3-way agreement. |
| **Homogeneous**, α = 0, r ≤ rCloud | `(4/3)πr³ρ_core`, rCore absent from the expression | identical | identical; requires rCore inert | **CONSISTENT** — and C's rCore-inertness invariant (I-6/I-7) is satisfied *by construction*: neither the α=0 density branch nor the α=0 mass branch references rCore. |
| **Power-law envelope**, rCore < r ≤ rCloud, α ∉ {0,−3} | `4πρ_core[rCore³/3 + (r^(3+α) − rCore^(3+α))/((3+α)rCore^α)]` | identical, 4 sites, cited Rahner+2018 Eq 25 | `(4π/3)ρ_core rCore³ + 4πρ_core rCore^(−α)(r^(3+α) − rCore^(3+α))/(3+α)` | **CONSISTENT** — algebraically identical (`1/rCore^α ≡ rCore^(−α)`). A confirmed numerically vs trapezoid (4.7e-12 rel); C confirmed vs adaptive quadrature (≤4e-16 rel) for α = 0, −1, −2, −2.5, −3, −3.5. **Both the `rCore^(−α)` normalisation and the additive core term are present** — C's S1 candidate S2-C-22 is verified *satisfied*, not violated. |
| **Envelope at α = −3 exactly** | **NaN** (`0/0`: numerator `r⁰−rCore⁰ = 0`, denominator `0·rCore⁻³ = 0`), RuntimeWarning only | no caveat in `mass_profile`; `powerLawSphere` raises ValueError "mass integral diverges" | `(4π/3)ρ_core rCore³ + 4πρ_core rCore³·ln(r/rCore)` | **INCONSISTENT** — the correct closed form is nowhere in the code. A and C **independently derived the identical log form**, and both verified it against direct quadrature (A: ratio 1.000000). The integral is **finite** — the uniform core regularises r→0 — so the code's own justification ("diverges") is false. See S2-R-06. |
| **Exterior**, r > rCloud | `mCloud + (4/3)πρ_ISM(r³ − rCloud³)` | identical | identical, and C called this "the highest-value single check in the slice" | **CONSISTENT** — C's S1 candidate S2-C-02 (fear of a clamp to `mCloud`) is **refuted**: the accumulating form is implemented for both profiles. Caveat: exactness of the `+mCloud` offset presumes `M(rCloud) = mCloud`, which is only true when the parameters are mutually consistent (see S2-R-03). |
| **Bonnor–Ebert, analytic path** | `mCloud·f_m(ξ(r))/f_m(ξ_out)`, ξ = ξ_out·r/rCloud | identical, claimed EXACT | identical; `m(ξ) = ξ²ψ′` *is* the integral because `d/dξ(ξ²ψ′) = ξ²e^{−ψ}` | **CONSISTENT and exact.** A verified by direct quadrature of 4πr²ρ_BE over [0, rCloud]: 1.000000e6 vs mCloud 1.000000e6 (−9.9e-9, quadrature error). |
| **Bonnor–Ebert, numerical fallback** | writes `M_arr[i]` with `i` the *inside-subarray* index; `M_arr[0] = 0.0` unconditionally; scalar r inside cloud → 0.0 | "falls back … trapezoidal, ~0.5 % error"; "r_arr must be sorted!" | must include the 0 → r[0] contribution; must not re-integrate at all | **INCONSISTENT** — three separate defects, all corroborated. See S2-R-04. |

### 2.2 The overriding caveat — two different ρ

Every "CONSISTENT" above is consistency with the **sharp** (unsmoothed) density. The density
`get_density_profile` actually returns is blended into `nISM` by a `tanh` ramp of half-width
`0.01·rCloud` centred on `rCloud`, and the mass closed forms never see it (`rho_arr` is passed into
`compute_enclosed_mass` and discarded on the densPL path). So:

- **Inside r ≲ 0.98·rCloud and outside r ≳ 1.03·rCloud, M is the exact integral of the returned ρ**
  (A measured `4πr²ρ_code / (dM/dr)` = 1.0000 at 0.5 rCloud and 1.0000 at 1.5 rCloud).
- **In a ~±2 % annulus around rCloud it is not**: ratio 0.7337 at 0.995 rCloud, **27.36** at
  1.005 rCloud.
- Integrated to rCloud, `∫4πr²ρ_smooth dr` vs `M_code(rCloud) = mCloud` differs by **−1.03 %**
  (α = 0), **−0.37 %** (α = −2), **−0.57 %** (densBE, Ω = 5) — i.e. **4–10× the code's own
  `MASS_TOLERANCE = 0.001`**.

B predicted exactly this from the prose alone (S2-B-10, S2-B-27) without seeing a number; C's
invariant I-1 (`dM/dr = 4πr²ρ` everywhere) is the thing being violated. **Three-lens corroboration
on the single most consequential finding in the slice.**

One nuance worth recording, because it partially rescues the documentation: the docstring claim
"mass conservation holds to O(SMOOTH_FRAC²)" is defensible for the mass integrated **across the
whole blend band** (the tanh is antisymmetric in linear r about rCloud, so the interior deficit
cancels the exterior excess to leading order). It is **not** true of `M(rCloud)` — the quantity the
validator, the shell mass and the gravity term all use — where only half the ramp has been
traversed and the error is linear in `SMOOTH_FRAC` (≈1 %, matching A's measurement). So the claim
is not a lie, it is a claim about a different integral than the one that matters.

### 2.3 Summary verdict

> **The coded `M(r)` is the exact integral of the coded sharp `ρ(r)` on every branch except
> α = −3 (NaN, no log branch) and the Bonnor–Ebert numerical fallback (indexing + dropped core
> mass). It is *not* the integral of the density `get_density_profile` actually returns, in a
> ~2 %-wide annulus at the cloud edge, where the local discrepancy reaches 27× and the integrated
> discrepancy 0.4–1.0 %.**

Additional corroborated consistencies worth banking (all three lenses, no defect):
`ρ` continuous at rCore by construction; the fractional-rCore shape factor
`K(f,α) = g = f³/3 + (1−f^(3+α))/((3+α)f^α)` is identical in all three accounts and is > 0 for every
f ∈ (0,1), α ≠ −3; `ρ = n·mu_convert` is a single r-independent multiplication whose numeric value
(A: 2.938e55 × 1.1783e-57 = 0.034618 M⊙pc⁻³ per cm⁻³) matches C's independently derived 0.0346101
to 0.02 % — within the m_H/M⊙ convention spread C itself flags.

---

## 3. Invariants (C's list, evaluated against A's transcription)

| # | Invariant | Verdict | Basis |
|---|---|---|---|
| I-1 | `dM/dr = 4πr²ρ` at every r | **FAIL** in the smoothing band (0.73× at 0.995 R, 27.4× at 1.005 R); PASS elsewhere | A measured; B predicted; C required |
| I-2 | `M(0) = 0`, `M → (4π/3)ρ_core r³` | **PASS** analytic paths (core branch and BE `m(0)=0`); **FAIL** in the BE fallback, which forces `M_arr[0]=0` for any `r_arr[0]` | A |
| I-3a | `ρ ≥ 0` everywhere | **PASS** (positive densities, tanh weight ∈ [0,1], positive base) | A |
| I-3b | `ρ` non-increasing | **PASS** for α ≤ 0 with nEdge ≥ nISM enforced; **NOT ENFORCED** for α > 0 — no validator rejects a positive exponent (C says it must) | A + C |
| I-3c | `m(ξ)`, `ψ(ξ)` strictly increasing (BE) | **PASS** — A re-solved: ρ/ρ_c strictly decreasing, m strictly increasing apart from 167 machine-precision ties at ξ < 2.9e-7, which the code's `np.unique` absorbs by design | A |
| I-3d | `ρ` continuous at rCore | **PASS** — automatic, `(rCore/rCore)^α = 1`; 3-way agreement | A, B, C |
| I-4 | `M(rCloud) = mCloud` | **PASS** (2e-16) for the sharp profile when rCloud came from the inverter; **FAIL** against the returned ρ (0.37–1.03 %); **FAIL** (1.63 %) whenever rCloud < rCore | A |
| I-5a | `rCloud(M(r)) = r` round-trip (PL) | **PASS** — both branches are exact algebraic inverses with a `rel_err > 1e-6 → RuntimeError` self-check. C's feared root-finder failure modes (T-05) do not apply: **there is no root-finder** | A vs C |
| I-5b | `xi2r(r2xi(r)) = r` | **PASS** — c_s recovered to 0.0e+00 relative error; ξ(rCloud) = 4.075531 = stored ξ_out | A |
| I-5c | homogeneous inverse domain | **PARTIAL** — exact for M > 0; for M < 0 it returns a **complex** number that raises an uncaught `TypeError` downstream | A |
| I-6 | α = 0 reduces to homogeneous | **PASS** — dedicated α==0 branches in both density and mass | A |
| I-7 | rCore inert at α = 0 | **PASS** — rCore does not appear in either α==0 branch | A |
| I-8 | Branch continuity in α through −3 | **FAIL at ε = 0** (NaN); conditioning of the general form is fine to \|3+α\| ≈ 1e-4 (3.8e-13 rel) and degrades to 1.4e-4 only at 1e-13 | A |
| I-9 | BE constants self-consistent | **PASS** numerically — A re-solved and got ξ_crit 6.4504, m 15.703, Bonnor coeff 1.1822 vs C's 6.4507514 / 15.704374 / 1.1822266. Three of the four constants are never referenced | A + C |
| I-10 | Ω ↔ ξ_out round-trip | **PASS** — `f_xi_from_rho(1/14.04)` = 6.45037 vs `XI_CRITICAL` 6.451 | A |
| I-11 | two Ṁ paths agree | **PASS** with each other (both use the smoothed ρ), **FAIL** against `dM/dt` of the returned M | A |
| I-12 | numerical == analytic enclosed mass | **FAIL** on the BE fallback path (see I-2) | A + C |
| I-13 | `ρ/n` constant, = μ·m_H | **PASS** — single multiplication; value matches C's derivation to 0.02 % | A + C |
| I-14 | scalar-in/scalar-out shape transparency | **NOT DETERMINABLE** — no lens tested the actual return type; B records the docstring's `<class 'float'>` claim, C flags `np.float64`/0-d as the classic blind spot | B + C |
| I-15 | exterior mass keeps growing | **PASS** — the accumulating form is implemented (C's top-priority check clears) | A + B + C |

---

## 4. Divergence table

| Item | A (does) | B (claims) | C (should) | Divergence | Resolution |
|---|---|---|---|---|---|
| Smoothed ρ vs sharp M | M is sharp; ρ returned is tanh-blended | prose says both "step" and "bridge"; claims Ṁ is EXACT and M analytic | requires `dM/dr = 4πr²ρ` | **AC** (+ internal B) | **Code defect** (S2-R-01/02) |
| α = −3 | NaN, no guard, no log branch | one module raises "diverges", another documents no caveat | log branch, integral is finite | **ABC** | **Code defect + false justification** (S2-R-06) |
| Exterior mass | accumulating | accumulating | accumulating | **none** | **Verified consistent — C's S2-C-02 dropped** |
| Envelope normalisation `rCore^(−α)` + core term | present | present | required | **none** | **Verified consistent — C's S2-C-22 dropped** |
| Isothermal LE ODE `u''+2u'/ξ = e^{−u}` | present, re-solved | documented | required | **none** | **Verified consistent — C's S2-C-08 dropped** |
| Series ICs ξ²/6 − ξ⁴/120 + ξ⁶/1890 | present, mutually consistent | documented | derived identically | **none** | **Verified consistent — C's S2-C-09 dropped** |
| BE critical constants (values) | re-computed, match | documented, mislabelled | computed | **BC** on the *label* only | Doc-drift (S2-R-18) |
| `T_eff` with γ | γ applied both ways; round-trip exact | "effective **isothermal** temperature" | γ must not appear | **ABC** | Code defect confined to the reported T (S2-R-07) |
| Ω > 14.04 | warning only, run proceeds | "must be < 14.04" (rejection language) | flag as unstable, do not reject | **AB**, C supports A | **Stale prose** (S2-R-17) |
| `nEdge` semantics | validators store/print `n_inside(rCloud)`; profile returns (nEdge+nISM)/2 | "n = nISM for r > rCloud" (sharp) | nEdge ≥ nISM is the guard against a density inversion | **AB** | Code/doc mismatch (S2-R-08) |
| BE outside rCloud | same tanh blend to nISM | interpolator "already handles" it | — | **none** | **B's S2-B-08 refuted by A — dropped** |
| BE smoothed at all? | yes, same ramp | BE branch has no blend comment | — | **none** | **B's S2-B-09 half refuted — dropped** |
| α < −3 fixed-rCore branch | `rhs ≤ 0` guard **is** the correct and complete condition; only the message is wrong | undocumented failure mode feared | — | **AB** on the message | **Demoted to S4 (S2-R-24)** |
| `compute_minimum_rCore` at α = 0 | explicit α==0 branch exists | explicit α==0 branch documented | feared `−1/0` | **none** | **C's S2-C-07/T-09 refuted — dropped** |
| root-finding rCloud | no root-finder; exact closed forms | "no root-finding" | closed form is the reference | **none** | **C's S2-C-05 root-find concerns dropped** |
| `odeint` vs `solve_ivp` arg order | code's own tabulated ξ_crit/Ω match the true values | `(y, t)` order noted | transposed args give plausible-but-wrong | **none** | **C's S2-C-29 refuted by A — dropped** |
| stale "v1" BE implementation | only one BE implementation in the slice | "CORRECT VERSION v2" scar language | — | **none** | **B's S2-B-28 dropped (rhetoric only)** |
| "mCloud should be in Msun" hedge | dimensional audit found no imbalance | hedge stated twice | — | **none** | **B's S2-B-24 dropped** |
| μ for ρ↔n | `mu_convert` = 1.4·m_H in M⊙; correct value | docstrings say both "(=1.4)" and "[M⊙]" | μ_H = 1.4 dimensionless × m_H | **AB** (same physics, contradictory prose + wrong default args) | Doc-drift + latent default (S2-R-11) |
| mass-consistency gate | tautology; cannot fail; mCloud ≤ 0 disables it | documented as a real constraint | each check must be independently able to fail | **AC** | **Code defect** (S2-R-05, S2-R-10) |
| tanh smoothing itself | present | documented, motivated (LSODA stall) | absent from the spec | scope note, not scope creep | Deliberate, documented numerical device; its *consequences* are what diverge |

---

## 5. Validation logic reconciled (documented-as-supported vs actually-rejected)

Lens B recorded what the prose says is supported; Lens A recorded what the validators do. Cross-checked:

| Parameter / value | Documented as supported (B) | Actually (A) | Verdict |
|---|---|---|---|
| `densPL_alpha = −3` | `mass_profile` documents the (3+α) formula with **no caveat**; `powerLawSphere` *module* docstring says "α ≠ 0" only | `compute_rCloud_powerlaw` **raises** (\|3+α\| < 1e-14); `compute_enclosed_mass_powerlaw` **accepts and returns NaN** | **Documented supported in 2 places, rejected in 1, silently NaN in a 3rd.** S2-R-06 |
| `rCore = None` for densPL, α ≠ 0 | `compute_rCloud_powerlaw` advertises a `rCore_fraction = 0.1` fallback | `_validate_powerlaw` **raises** when rCore is None with α ≠ 0 | **Advertised in one module, denied by the validator.** S2-R-23 |
| `mCloud ≤ 0` (densPL) | mass consistency documented as a hard constraint | `mass_error` forced to 0.0 when mCloud ≤ 0 ⇒ test cannot fire; `create_BE_sphere` and `validate_mass_at_rCloud` both reject it | **Accepted by the GMC validator, rejected by two siblings.** S2-R-10 |
| `Ω > 14.04` | "must be < 14.04 for stability" (rejection language) | warning only; runs proceed. C agrees warning is correct | **Prose is stale, code is right.** S2-R-17 |
| `Ω == 14.04` exactly | — | `is_stable = Ω < 14.04` → False, but the log warning uses `Ω > 14.04` → silent | Boundary conventions disagree by one point. S2-R-17 |
| `rCore ≥ rCloud` | called a "pathological case" (clamped only inside `compute_minimum_rCore`) | **accepted everywhere else**; C says reject | **Accepted by validators, unhandled downstream** (validator and mass profile then take different branches). S2-R-03 |
| `densPL_alpha > 0` | no α > 0 branch described anywhere | accepted; `compute_minimum_rCore`'s bound inverts sense; masked by a clip | **Accepted, unhandled; C says reject outright.** S2-R-09 |
| `Ω` above ~222 | no ceiling documented | `target_rho_rhoc < rho_rhoc[-1]` → **ValueError** (fails loudly, does not clamp) | Code satisfies C's requirement; only the doc is missing. S2-R-19 |
| `nEdge ≥ nISM` | hard constraint on the cloud edge density | constrains `n_inside(rCloud)`, a value the profile never returns (tanh gives (nEdge+nISM)/2) | Guard applies to the wrong quantity. S2-R-08 |
| module usage example `check_gmc_constraints(rCloud=150, nEdge=0.5, mCloud=1e5, M_computed=1.001e5)` | advertised as a working invocation | runs (nISM defaults to 1.0) but returns **two** errors: 0.5 < 1.0 edge-density, and \|1.001e5−1e5\|/1e5 = 1.0000000000001455e-3 > 1e-3 | The module's own example fails its own validator. S2-R-20 |

---

## 6. Dropped or demoted (filter log)

Dropped as **verified consistent** (a lens's concern is answered by another lens's evidence):
S2-C-02 (exterior clamp — accumulating form is implemented), S2-C-22 (envelope normalisation and
core term both present), S2-C-08 (isothermal LE ODE is the coded one), S2-C-09 (series ICs present
and mutually consistent), S2-C-13 (BE analytic mass is `m(ξ)/m(ξ_out)`, no quadrature),
S2-C-17 (r↔ξ round-trip exact), S2-C-06/S2-C-20 (ρ/n is a single r-independent constant matching
C's derived value to 0.02 %), S2-C-10/11/12 (constant *values* correct — label issue survives as
S2-R-18), S2-C-28 (α = 0 reduction and rCore inertness hold by construction).

Dropped as **refuted**: S2-C-29 (transposed ODE args — the code's own ξ_crit/Ω agree with the true
values), S2-C-07/T-09 (α = 0 division by zero — an explicit α==0 branch exists, per both A and B),
S2-C-05 root-finding failure modes (there is no root-finder), S2-B-08 (BE outside rCloud —
same tanh hand-off to nISM), S2-B-09's "densBE is not smoothed" half (it is), S2-B-28 (stale v1 BE
implementation — only one exists in the slice), S2-B-24 ("mCloud should be in Msun" hedge — A's
dimensional audit found no imbalance).

**Demoted**: S2-B-06 (α < −3 fixed-rCore branch) S3 → S4 — A shows the `rhs ≤ 0` guard is the
correct and complete condition for that regime, so the feared NaN cannot occur; only the error
*message* is wrong. S2-B-15 (undocumented Ω ceiling) S3 → S4 — it raises rather than clamps.
S2-B-09 (C^∞ claim) S3 → S4, with the reason corrected (the claim fails at rCore for α ≠ 0 and for
the BE cubic interpolant, not because densBE is unsmoothed).

Input candidates: 17 (A) + 28 (B) + 30 (C) = **75** → **31** reconciled items.

---

## 7. Merged ranked findings

```json
[
  {
    "id": "S2-R-01",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 214,
    "class": "numerical",
    "severity": "S2",
    "claim": "M(r) is not the integral of the rho(r) the code returns. get_density_profile blends the cloud density into nISM with a tanh ramp of half-width 0.01*rCloud centred on rCloud; compute_enclosed_mass receives that smoothed rho_arr and, on the densPL path, discards it in favour of a closed form that integrates the SHARP profile (the densBE analytical path does the same). The resulting M(rCloud) disagrees with the integral of the returned density by 4-10x the code's own MASS_TOLERANCE.",
    "evidence": "A transcribed density_profile.py:128-130 (delta = 0.01*rCloud, w = 0.5(1+tanh((r-rCloud)/delta)), n = n_inside(1-w) + nISM*w) and mass_profile.py:260 routing densPL to compute_enclosed_mass_powerlaw(r_arr, params), which never touches rho_arr. A measured: alpha=0, mCloud=1e6, nCore=1e3 cm^-3, rCloud=19.034 pc -> trapezoid(4 pi r^2 rho_smooth) = 9.897358e5 vs M_code(rCloud) = 1.000000e6, rel -1.0264%; alpha=-2, rCore/rCloud=0.1 -> -0.3677%; densBE Omega=5 -> -0.5724%. The same quadrature on the SHARP profile reproduces the closed form to 4.7e-12, confirming which density the closed form integrates. B predicted the same mismatch from prose alone (S2-B-10, S2-B-27) without a number. C's invariant I-1 requires dM/dr = 4 pi r^2 rho at every r.",
    "expected": "Either integrate the smoothed density that is actually returned, or apply the same smoothing to M(r); either way M(rCloud) must reproduce mCloud to better than MASS_TOLERANCE = 1e-3. Note the documented 'mass conservation holds to O(SMOOTH_FRAC^2)' is true only for the mass integrated across the whole blend band (the tanh is antisymmetric about rCloud); at r = rCloud itself the error is linear in SMOOTH_FRAC, which is exactly what A measured.",
    "failure_scenario": "The shell's swept-up mass and the gravitational mass interior to it come from a profile 0.4-1.0% inconsistent with the density used for ram pressure and cooling. The bias grows as the shell approaches the cloud edge - precisely where the transition to breakout is decided - and the code's own mass-consistency validator cannot see it (S2-R-05).",
    "repro": "Reimplement get_density_profile (density_profile.py:105-170) and compute_enclosed_mass_powerlaw (mass_profile.py:297-344) standalone in code units; compare scipy.integrate.trapezoid(4*pi*r**2*rho_smooth, r) over [0, rCloud] against the closed form at rCloud. Repeat with SMOOTH_FRAC halved: if the discrepancy halves (rather than quartering) the O(SMOOTH_FRAC^2) claim does not apply at rCloud.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S2-A-01", "S2-B-10", "S2-B-27", "S2-C-03", "S2-C-04"]
  },
  {
    "id": "S2-R-02",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 224,
    "class": "numerical",
    "severity": "S2",
    "claim": "Same root cause as S2-R-01, different observable: dM/dt returned by get_mass_profile(return_mdot=True) and by compute_mass_accretion_rate is 4*pi*r^2*rho_SMOOTHED*rdot, which is not the time derivative of the M(r) returned alongside it. The two disagree by a factor 27 just outside rCloud and by 27% just inside.",
    "evidence": "A: mass_profile.py:224 and :479 use rho from get_mass_density (smoothed) while M comes from the sharp closed form. Measured on a valid alpha=-2 config (mCloud=1e6, nCore=1e4 cm^-3, rCore=2.909 pc, rCloud=29.095 pc, nEdge/nISM=100), ratio (4 pi r^2 rho_code)/(dM/dr from M): r/rCloud = 0.500 -> 1.0000; 0.970 -> 0.9976; 0.995 -> 0.7337; 1.005 -> 27.36; 1.030 -> 1.2306; 1.500 -> 1.0000. B (S2-B-10) derived the same inconsistency from the docstrings, noting the prose bounds the integrated mass error but never the local derivative error. C (I-11, S2-C-15) requires both paths to equal 4 pi r^2 rho(r) rdot with the same local rho used by M.",
    "expected": "Ratio 1.0 at every radius including the cloud edge - one density must feed both M and Mdot.",
    "failure_scenario": "During cloud breakout the shell mass integrated from Mdot diverges from the M(r) used in the momentum/energy equations, so mass is effectively created or destroyed at the cloud edge. A shell stalling near rCloud sees a 27x overestimate of accreted mass just outside the edge, which can spuriously halt or reverse expansion.",
    "repro": "Central-difference the coded M(r) at r/rCloud in {0.5, 0.97, 0.995, 1.005, 1.03, 1.5} and compare to 4*pi*r^2*get_density_profile(r)*mu_convert.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S2-A-02", "S2-B-10", "S2-C-15", "S2-C-03"]
  },
  {
    "id": "S2-R-03",
    "file": "trinity/cloud_properties/powerLawSphere.py",
    "line": 148,
    "class": "regime",
    "severity": "S2",
    "claim": "rCloud < rCore is silently reachable (the fixed-rCore inversion never checks rCloud > rCore), and when it happens the validator and the mass profile evaluate different branches for M(rCloud): the validator reports a 2e-16 mass error while the profile the solver integrates is off by 1.6%.",
    "evidence": "A: compute_rCloud_powerlaw (powerLawSphere.py:148-178) rejects only rhs <= 0. With alpha=-2, rCore=5 pc, nCore=1e4 cm^-3 (uniform-core mass 1.8126e5 Msun) and mCloud = 1.4501e5 Msun (0.8x the core mass) the inversion returns rCloud = 4.667 pc < rCore = 5 pc. _validate_powerlaw:435 then evaluates the region-2 closed form unconditionally -> M_computed = 1.450117e5, error 2.01e-16, validation PASSES; compute_enclosed_mass_powerlaw:326-327 instead selects the core branch -> M(rCloud) = 1.473748e5, error 1.63e-2, 16x MASS_TOLERANCE. C independently predicted this exact failure mode (S2-C-05(ii): 'if M_cloud is below the core mass the root lies inside the core and the objective uses the wrong branch') and its validity table requires rejecting r_core >= r_cloud. B records only a 'pathological case' clamp, and that clamp lives in a different function (compute_minimum_rCore).",
    "expected": "Reject rCore >= rCloud (a core larger than the cloud is meaningless), or fall back to the homogeneous inversion when mCloud <= the core mass, as C prescribes. The validator should evaluate M_computed through get_mass_profile so it sees the same branch selection the solver does.",
    "failure_scenario": "A dense, compact .param (small mCloud with a large rCore) validates clean and then runs with a total cloud mass 1.6% below the requested mCloud, with the whole cloud silently uniform rather than power-law.",
    "repro": "alpha=-2, rCore=5.0 pc, nCore=1e4 cm^-3, mCloud=1.4501e5 Msun: compare the region-2 formula at rCloud (validator) with compute_enclosed_mass_powerlaw(rCloud).",
    "confidence": "high",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S2-A-04", "S2-C-05", "S2-B-25"]
  },
  {
    "id": "S2-R-04",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 415,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The Bonnor-Ebert numerical fallback is broken three ways: it indexes the output array with the inside-subarray index (writing enclosed masses into the wrong slots), it forces M_arr[0] = 0.0 so the mass interior to r_arr[0] is dropped and a scalar r inside the cloud returns 0.0, and the documented 'must be sorted!' precondition on r_arr is unenforced. It fails silently rather than raising.",
    "evidence": "A executed the loop body with r_arr = [2, 5, 12, 8], rCloud = 10: inside mask [T, T, F, T]; the loop writes flat slots 0, 1, 2 - slot 2 belongs to r=12 (outside, later overwritten) and slot 3 (r=8 pc, inside the cloud) is left at 0.0. A also verified a scalar r inside the cloud returns exactly 0.0. C (S2-C-14) independently predicted the dropped 0 -> r_arr[0] contribution ('0.1% at alpha = 0 but 3.6% at alpha = -2 with rCore_fraction = 0.1, i.e. above MASS_TOLERANCE'). B (S2-B-13) recorded the unenforced sort precondition and (S2-B-02) that the fallback's documented ~0.5% error is 5x the 0.1% tolerance it must satisfy. The path is taken only when densBE_f_m / densBE_xi_out are absent, which create_BE_sphere_from_params guarantees - so it is latent.",
    "expected": "Assign through the boolean mask (M_arr[inside_cloud] = <cumulative integral>), include the [0, r_inside[0]] contribution analytically, and either sort internally or raise on unsorted input. Better, per C: use m(xi) = xi^2 psi' and never re-integrate.",
    "failure_scenario": "Any densBE call path whose params lack densBE_f_m/densBE_xi_out - a restart, a reader-reconstructed params dict, an output-analysis helper - silently returns zero or misplaced enclosed masses instead of raising.",
    "repro": "Run the loop body of mass_profile.py:415-422 on r_arr=[2,5,12,8], rho_arr=linspace(5,1,4), rCloud=10 and inspect M_arr; separately call with a scalar r inside the cloud.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S2-A-05", "S2-B-02", "S2-B-13", "S2-C-14", "S2-C-13"]
  },
  {
    "id": "S2-R-05",
    "file": "trinity/cloud_properties/validate_gmc.py",
    "line": 435,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The 'mass consistency' gate in check_gmc_constraints is a tautology in both profile branches - M_computed is recomputed with the same closed form that was algebraically inverted to produce rCloud - so it can never fail, and it provides no assurance about the mass profile the solver actually integrates. The one validator that would catch the real errors (validate_mass_at_rCloud, the only one that calls get_mass_profile) is never called.",
    "evidence": "A: _validate_powerlaw:416 obtains rCloud from compute_rCloud_powerlaw (which itself already asserts rel_err <= 1e-6 at powerLawSphere.py:168), then line 435 re-evaluates the identical region-2 expression; measured mass_error 2.0e-16 (alpha=-2), exactly 0 (alpha=0). _validate_bonnor_ebert:507 computes M = 4 pi m_dim rho_core a^3 where a came from solving that same equation for c_s, so it also returns mCloud identically. Meanwhile the profile the solver integrates is off by 0.4-1.0% (S2-R-01) or 1.6% (S2-R-03), invisible to this gate. C (S2-C-23, section 6) requires all three constraints to be 'each independently able to fail' and the mass residual reported as a number. B records the gate as a genuine constraint in five places.",
    "expected": "M_computed should come from get_mass_profile(rCloud, params) - the same code path the solver uses - as validate_mass_at_rCloud (mass_profile.py:525) already does.",
    "failure_scenario": "Operators read 'mass error 0.0000%' and trust that M(rCloud) == mCloud, while the integrated profile is off by up to 1.6%. Both S2-R-01 and S2-R-03 are invisible to the project's own headline validation.",
    "repro": "Instrument _validate_powerlaw to also print get_mass_profile(rCloud, params) and compare with M_computed for the rCloud<rCore config in S2-R-03.",
    "confidence": "high",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S2-A-07", "S2-C-23", "S2-C-04"]
  },
  {
    "id": "S2-R-06",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 334,
    "class": "divergence",
    "severity": "S3",
    "claim": "densPL_alpha = -3 makes the region-2 mass expression 0/0 and returns NaN for the entire envelope, with no guard and no exception - only a numpy RuntimeWarning. The correct closed form is logarithmic and finite; the code's own rejection message in powerLawSphere ('mass integral diverges') is false, because the uniform core regularises r -> 0.",
    "evidence": "Three-lens agreement, the strongest in the slice. A executed it: alpha=-3, rCore=1 pc, rCloud=20 pc gives M_code([0.5, 5, 10, 25]) = [1.813e+02, nan, nan, 1.001e+06] - finite in the core and the ISM, NaN across the whole envelope. A and C INDEPENDENTLY derived the identical correct form M(r) = 4 pi rho_core [rCore^3/3 + rCore^3 ln(r/rCore)] for rCore < r <= rCloud; A verified it against direct quadrature (log form 1.146717e4 vs numerical 1.146717e4, ratio 1.000000), C against adaptive quadrature (<=4e-16 relative). B records that mass_profile documents the (3+alpha) denominator with NO alpha = -3 caveat, that powerLawSphere.py:78/:142 raises ValueError 'mass integral diverges', and that the powerLawSphere module docstring is a third variant stating only 'alpha != 0'.",
    "expected": "Branch on |3+alpha| < tol (any tol in 1e-10..1e-6 is safe; A measured the general form's conditioning as 3.8e-13 rel err at |3+alpha| = 1e-4 degrading to 1.4e-4 at 1e-13; C measured 1.7e-6 at 1e-6 and 4.5e-7 at 1e-12) and use the log form. The inversion also has an exact log closed form: rCloud = rCore*exp(A/rCore^3) with A = M/(4 pi rhoCore) - rCore^3/3. At minimum, correct the 'diverges' message and guard the mass function.",
    "failure_scenario": "A NaN enclosed mass propagates silently into the shell mass, the gravity term and every downstream force. Reachability is gated: compute_rCloud_powerlaw raises for |3+alpha| < 1e-14, so the normal setup path is safe; the exposure is any path that supplies rCloud externally (initial_profile reconstruction from metadata, a hand-built params dict, a reader). C also notes the .param schema declares -2 <= alpha <= 0, which - if enforced on every entry path - closes the remaining exposure.",
    "repro": "compute_enclosed_mass_powerlaw with densPL_alpha=-3.0, rCore=1.0, rCloud=20.0 on r = [0.5, 5, 10, 25]; compare 4*pi*rhoCore*(rCore**3/3 + rCore**3*log(r/rCore)) against trapezoid(4*pi*r**2*rho_sharp, r).",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "ABC",
    "status": "corroborated",
    "source_ids": ["S2-A-03", "S2-A-08", "S2-B-05", "S2-C-01"]
  },
  {
    "id": "S2-R-07",
    "file": "trinity/cloud_properties/bonnorEbertSphere.py",
    "line": 431,
    "class": "coefficient",
    "severity": "S3",
    "claim": "densBE_Teff is defined through an adiabatic sound speed (T = mu*m_H*c_s^2/(gamma*k_B), gamma default 5/3) although the sphere solved is the ISOTHERMAL Lane-Emden equation, for which P = rho c_s^2 and c_s^2 = k_B T/(mu m_H) with no gamma. The stored temperature is therefore a factor gamma = 5/3 below the temperature that reproduces the sphere's pressure.",
    "evidence": "A: bonnorEbertSphere.py:200-203 integrates u'' + 2u'/xi = exp(-u) (isothermal), line 431 sets T_eff = mu*MSUN_TO_G*c_s^2/(gamma*K_B_CGS). For Omega=5, mCloud=1e6, nCore=1e3 cm^-3 the code reports c_s = 8.968 km/s with T_eff = 8189 K; the isothermal temperature matching that c_s is 13648 K. B: the field is documented as 'Effective isothermal temperature [K]' while the formula carries gamma, and the module is titled isothermal throughout. C derived from first principles that gamma plays no role in the hydrostatic structure and that T = mu m_H c_s^2/k_B is the only consistent back-solve. IMPORTANT MITIGATION from A: r2xi applies the same gamma, so the round trip is exact (c_s recovered to 0.0e+00 relative error, xi(rCloud) = 4.075531 = stored xi_out) - the PROFILE is unaffected. C's alternative worry that gamma silently rescales r0 and hence rCloud by sqrt(5/3) is therefore refuted.",
    "expected": "Use T = mu*m_H*c_s^2/k_B (gamma = 1) for the isothermal structure, or rename the quantity and document the gamma convention so every consumer applies it identically.",
    "failure_scenario": "Any module outside this slice that reads densBE_Teff as a gas temperature - thermal pressure n k T, cooling-table lookup, initial shell temperature, cloudy export - is low by 5/3. C adds a second, single-lens concern: the thermal mu for a cold molecular cloud is ~2.33, not the 1.4 used for rho<->n, which would compound the error by a further 1.66x; neither A nor B can confirm which mu the temperature conversion uses beyond 'the same one'.",
    "repro": "For any densBE run compare params['densBE_Teff'] with mu_convert*Msun2g*(densBE_sigma*1e5)**2/k_B; the ratio should be exactly gamma_adia. Then grep for every consumer of densBE_Teff outside cloud_properties.",
    "confidence": "medium",
    "lenses": ["A", "B", "C"],
    "divergence": "ABC",
    "status": "corroborated",
    "source_ids": ["S2-A-10", "S2-B-04", "S2-C-16"]
  },
  {
    "id": "S2-R-08",
    "file": "trinity/cloud_properties/density_profile.py",
    "line": 130,
    "class": "state",
    "severity": "S3",
    "claim": "The nEdge that validators compute, store in params and print is never the density the profile returns at rCloud: because the tanh ramp is centred on rCloud, get_density_profile(rCloud) = (nEdge + nISM)/2 exactly. The nEdge >= nISM constraint therefore guards a quantity the profile does not produce.",
    "evidence": "A: at r = rCloud, tanh(0) = 0 so w = 0.5 and n = 0.5*n_inside + 0.5*nISM (density_profile.py:130, 148, 164), while validate_gmc.py:419 sets nEdge = nCore*(rCloud/rCore)**alpha and bonnorEbertSphere.py:575 sets nEdge = n_core/Omega, i.e. n_inside(rCloud) in both cases. B independently found the prose contradiction: get_density_profile's Notes and the BE section both state 'n = nISM for r > rCloud' unqualified while the module docstring says that step is replaced by a bridge (S2-B-08/S2-B-09). C's spec has a sharp edge and makes nEdge >= nISM the guard against a density inversion (S2-C-19).",
    "expected": "Either report the evaluated n(rCloud), or centre the ramp at rCloud + delta so the profile actually attains n_inside(rCloud) at rCloud, so that nEdge means what it is checked to mean.",
    "failure_scenario": "A config tuned to nEdge == nISM (the validator's inclusive boundary) actually has n(rCloud) = nISM and passes; a config with nEdge = 2*nISM has n(rCloud) = 1.5*nISM. Any downstream code reading params['nEdge'] as the ambient density the shell meets at breakout is off by up to a factor 2.",
    "repro": "get_density_profile(rCloud, params) vs params['nEdge'].value for any densPL config.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S2-A-09", "S2-B-08", "S2-B-09", "S2-C-19"]
  },
  {
    "id": "S2-R-09",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 614,
    "class": "sign",
    "severity": "S3",
    "claim": "A positive densPL_alpha is accepted everywhere and handled correctly nowhere. compute_minimum_rCore's rCore_min is a lower bound only for alpha < 0; for alpha > 0 the inequality reverses and rCore_min becomes an UPPER bound, so multiplying by margin=1.1 moves away from the constraint. No validator rejects alpha > 0, which C says is not a cloud at all (density increasing outward).",
    "evidence": "A: the constraint nCore*(rCloud/rCore)^alpha >= nISM rearranges to rCore >= rCloud*(nCore/nISM)^(1/alpha) for alpha < 0 but rCore <= ... for alpha > 0, while line 614-618 computes it unconditionally. Verified: alpha=-2 gives rCore_min=0.2 with nEdge/nISM = 0.81/1.00/1.21 at margin 0.9/1.0/1.1 (correct); alpha=+1 gives 200000 pc and alpha=+0.5 gives 2e9 pc for a 20 pc cloud. B: the docstring derivation is guarded by 'For alpha < 0' and no alpha > 0 branch is described anywhere. C's validity table: 'reject alpha > 0 (density increasing outward - not a cloud)'.",
    "expected": "Reject alpha > 0 at the validator (C), or branch on sign(alpha) and document that the helper is defined only for alpha <= 0.",
    "failure_scenario": "Masked for the suggested radius: the huge rCore_min trips the 'rCore_suggested >= rCloud' clip and falls back to 0.9*rCloud, which does satisfy nEdge >= nISM. The returned fourth value rCore_min is nonsense, and a caller using it as a hard floor gets a meaningless number. The broader exposure is that an inverted cloud (alpha > 0) runs to completion. Severity depends on the .param schema range, which no lens could see: C reports the spec declares -2 <= alpha <= 0.",
    "repro": "compute_minimum_rCore(1e4, 1.0, 20.0, +1.0) -> rCore_min = 200000.0 pc for a 20 pc cloud; then check whether validate_gmc_params rejects alpha = +1.",
    "confidence": "medium",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S2-A-13", "S2-B-07", "S2-C-19"]
  },
  {
    "id": "S2-R-10",
    "file": "trinity/cloud_properties/validate_gmc.py",
    "line": 250,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "mCloud <= 0 silently passes GMC validation on the densPL path: mass_error is forced to 0.0 whenever mCloud <= 0, so the mass-consistency test cannot fire, and no other check rejects a non-positive cloud mass. mCloud < 0 additionally produces a complex rCloud and an uncaught TypeError.",
    "evidence": "A executed it: mass_error = abs(M_computed - mCloud)/mCloud if mCloud > 0 else 0.0 (line 250), then 'if mass_error > mass_tolerance'. check_gmc_constraints(rCloud=10.0, nEdge=nCore, mCloud=0.0, M_computed=1e9, nISM) returns errors []. For mCloud < 0 with alpha=0, compute_rCloud_homogeneous returns (3*-1e6/(4 pi rho))**(1/3) = 4.417+7.651j, which raises an uncaught TypeError at the 'rCloud > r_max' comparison (TypeError is not in the caught tuple, and the comparison is outside the try). The BE branch is saved only because create_BE_sphere:351 rejects M_cloud <= 0, and validate_mass_at_rCloud:528 also rejects it - so two siblings reject what this validator accepts.",
    "expected": "Reject mCloud <= 0 explicitly, as create_BE_sphere and validate_mass_at_rCloud already do.",
    "failure_scenario": "A typo or a sweep expression yielding mCloud = 0 passes validation and produces a zero-mass cloud; mCloud < 0 crashes with an opaque TypeError instead of a parameter error.",
    "repro": "check_gmc_constraints(10.0, nEdge, 0.0, 1e9, nISM) -> {'errors': [], 'mass_error': 0.0}.",
    "confidence": "high",
    "lenses": ["A"],
    "divergence": "AC",
    "status": "single-lens",
    "source_ids": ["S2-A-06", "S2-C-23"]
  },
  {
    "id": "S2-R-11",
    "file": "trinity/cloud_properties/powerLawSphere.py",
    "line": 51,
    "class": "units",
    "severity": "S3",
    "claim": "The mu=1.4 default argument in compute_rCloud_homogeneous, compute_rCloud_powerlaw, compute_consistent_params and create_BE_sphere is in the wrong unit system for the bodies that use it, and the docstrings contradict themselves about which convention applies: five sites say '(=1.4)', two unit blocks say '[Msun]', one admits '1.4 is a placeholder'. The module's own usage examples pass cm^-3-magnitude densities against arguments documented as [1/pc^3].",
    "evidence": "A: the bodies compute rhoCore = nCore*mu with nCore in pc^-3, so mu must be Msun per particle; every in-slice caller passes params['mu_convert'].value = 1.4*m_H*g2Msun = 1.1783e-57 Msun. A caller taking the default gets rCloud wrong by (1.4/1.1783e-57)^(1/3) ~ 1.06e19 (compute_rCloud_homogeneous(1e6, 1e3*ndens_cgs2au) -> 1.79e-19 pc vs 19.03 pc). B catalogued the contradiction (S2-B-01) and the cgs-magnitude usage examples (S2-B-16). C independently derived the correct constant: rho[Msun/pc^3] = 0.0346101*n[cm^-3] at mu_H = 1.4 per hydrogen nucleus - which matches A's inferred code-unit product 2.938e55 * 1.1783e-57 = 0.034618 to 0.02%. So the CODE'S ACTUAL CONVERSION IS CORRECT; the defect is confined to the default arguments and the prose.",
    "expected": "No default, or a default equal to the code-unit mu_convert; docstrings that say '(=1.4)' should say '[Msun], = 1.4 * m_H expressed in Msun'; the usage examples should use code-unit values or be labelled illustrative.",
    "failure_scenario": "A test, a tools/ utility or a notebook calls compute_rCloud_homogeneous(mCloud, nCore) without mu, or copies the validate_gmc docstring example verbatim, and silently gets a cloud radius wrong by ~1e19 (or many orders of magnitude) with no error raised - the validator then blames the cloud parameters.",
    "repro": "compute_rCloud_homogeneous(1e6, 1e3*ndens_cgs2au) with and without an explicit mu; run the validate_gmc.py:3 usage examples verbatim.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S2-A-11", "S2-B-01", "S2-B-16", "S2-C-06", "S2-C-20"]
  },
  {
    "id": "S2-R-12",
    "file": "trinity/cloud_properties/density_profile.py",
    "line": 56,
    "class": "state",
    "severity": "S3",
    "claim": "The documented 'Required keys' lists are incomplete for the densBE path: get_density_profile lists only densBE_f_rho_rhoc although it calls r2xi (which needs densBE_Teff, nCore, mu_convert, gamma_adia), and get_mass_profile's list omits densBE_f_m and densBE_xi_out, which the BE mass function needs.",
    "evidence": "B: density_profile.py:56 lists densBE_f_rho_rhoc only; density_profile.py:27 imports bonnorEbertSphere 'for r2xi conversion'; bonnorEbertSphere.py:583 documents r2xi's four required keys; mass_profile.py:352 says the analytical method needs densBE_f_m and densBE_xi_out. A confirms the call chain: get_density_profile's BE branch computes xi(r) = be_r2xi(r, params), and compute_enclosed_mass_bonnor_ebert takes its analytical path only when densBE_f_m/densBE_xi_out are present - otherwise it silently takes the broken fallback (S2-R-04).",
    "expected": "The required-keys lists should name every key actually read, so a caller constructing a params dict by hand knows what to supply.",
    "failure_scenario": "A hand-built densBE params dict following the docstring raises KeyError deep inside r2xi, or - worse - silently takes the broken numerical fallback because densBE_f_m is 'not available'.",
    "repro": "Build a params dict containing exactly the documented keys for densBE and call get_density_profile / get_mass_profile.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S2-B-11", "S2-A-05"]
  },
  {
    "id": "S2-R-13",
    "file": "trinity/cloud_properties/initial_profile.py",
    "line": 144,
    "class": "state",
    "severity": "S3",
    "claim": "The mock params dict built for the densBE reconstruction path seeds only four keys and is missing four that create_BE_sphere_from_params assigns unconditionally (densBE_xi_arr, densBE_u_arr, densBE_dudxi_arr, densBE_rho_rhoc_arr), so that path raises KeyError if it is reached.",
    "evidence": "A: initial_profile.py:144-147 seeds densBE_Teff, densBE_xi_out, densBE_f_rho_rhoc, densBE_f_m only; bonnorEbertSphere.py:565-568 does params['densBE_xi_arr'].value = ... for the four array keys with no membership guard, while only the three keys at :554-560 get the 'if key not in params' treatment. B corroborates both halves from prose: the documented side-effect list at :502 includes all four array keys, :552 is labelled a 'safety fallback for standalone usage', and initial_profile.py:143 says 'Pre-seed them so the .value = ... assignment lands cleanly'.",
    "expected": "Seed those four keys in the mock dict, or extend the key-creation loop at bonnorEbertSphere.py:554 to cover them.",
    "failure_scenario": "build_initial_cloud_profile(dens_profile='densBE', ...) - the post-hoc profile reconstruction used against metadata.json - dies with KeyError instead of returning (r, n, M).",
    "repro": "Call build_initial_cloud_profile with dens_profile='densBE' and check whether _init_bonnor_ebert_cloud reaches create_BE_sphere_from_params. NOTE: _init_bonnor_ebert_cloud lives in trinity/phase0_init/get_InitCloudProp.py, outside this slice, so reachability is unverified by any lens.",
    "confidence": "low",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S2-A-14", "S2-B-22"]
  },
  {
    "id": "S2-R-14",
    "file": "trinity/cloud_properties/bonnorEbertSphere.py",
    "line": 502,
    "class": "state",
    "severity": "S3",
    "claim": "create_BE_sphere_from_params documents a 10-key 'Updates params with:' list but performs at least two further writes that are not in it: a sigma [km/s] value derived from c_s, and unspecified 'derived cloud properties'.",
    "evidence": "B: the documented list is densBE_Teff, densBE_xi_arr, densBE_u_arr, densBE_dudxi_arr, densBE_rho_rhoc_arr, densBE_f_rho_rhoc, densBE_f_m, densBE_xi_out, rCloud, nEdge; comments at :564 ('c_s [cm/s] -> sigma [km/s]') and :573 ('Also update derived cloud properties') describe writes outside it. A corroborates the mechanism from the arithmetic: bonnorEbertSphere.py:564 computes c_s/1.0e5 -> km/s and :575 sets params['nEdge'] = n_core/Omega. Neither lens confirmed the destination key name for the km/s value.",
    "expected": "The documented side-effect list must name every mutated key.",
    "failure_scenario": "A .param that sets a turbulent velocity dispersion has it silently overwritten by the BE sphere's c_s when dens_profile='densBE', changing the physics with no log line and no mention in the documented contract.",
    "repro": "Snapshot params keys and values before and after create_BE_sphere_from_params and diff against the documented list.",
    "confidence": "medium",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S2-B-22", "S2-A-10"]
  },
  {
    "id": "S2-R-15",
    "file": "trinity/cloud_properties/initial_profile.py",
    "line": 3,
    "class": "state",
    "severity": "S3",
    "claim": "initial_profile claims to be 'the inverse of phase0_init/get_InitCloudProp.py' and asserts, without proof, that calling the constructor with post-correction scalars is a no-op for the auto-correction branches. Relatedly, nEdge is accepted as an input although it is fully determined by the other parameters, creating a second source of truth.",
    "evidence": "B: initial_profile.py:3 states the idempotency claim over three mutated keys (rCore, rCloud, nEdge) plus 'etc.'; :107 notes 'The phase-0 constructors mutate a few keys (rCore, rCloud, nEdge) via auto-correction'; :72 documents nEdge as 'BE only' while :126 says _init_powerlaw_cloud also populates it. C (S2-C-26) independently notes nEdge is derivable (nCore*(rCloud/rCore)^alpha for PL, nCore/Omega for BE) so accepting it as an argument needs a consistency check. A supplies a concrete non-idempotency candidate from the same package: compute_minimum_rCore applies a multiplicative margin of 1.1 to rCore.",
    "expected": "A round-trip test: for parameter sets that DO trigger an auto-correction on the first pass, reconstructing from the post-correction scalars must reproduce the original (r, n, m) arrays. nEdge should be derived internally or checked against the derived value.",
    "failure_scenario": "If any correction is not idempotent, every consumer that reconstructs arrays - plot scripts, the cloudy exporter - silently plots/exports a different cloud than was simulated, with no error raised.",
    "repro": "Run a config that triggers the nEdge < nISM correction; persist the phase-0 arrays; rebuild via build_initial_cloud_profile from metadata.json scalars; np.testing.assert_array_equal. Separately, pass a deliberately wrong nEdge (2x derived) and check whether it is used, ignored or rejected.",
    "confidence": "medium",
    "lenses": ["B", "C"],
    "divergence": "BC",
    "status": "corroborated",
    "source_ids": ["S2-B-19", "S2-B-20", "S2-C-26"]
  },
  {
    "id": "S2-R-16",
    "file": "trinity/cloud_properties/powerLawSphere.py",
    "line": 214,
    "class": "state",
    "severity": "S3",
    "claim": "compute_consistent_params is advertised as 'the recommended way to set up test parameters' but returns key names none of the slice's consumers use ('M_cloud', 'alpha', 'mu' instead of 'mCloud', 'densPL_alpha', 'mu_convert'), omits 'dens_profile' and 'nISM', and can return a triple whose nEdge is below nISM. It is called nowhere in the slice and its nISM default is in the wrong unit system.",
    "evidence": "B: the documented return keys are 'rCloud','rCore','nEdge','M_cloud','nCore','alpha','mu' while get_density_profile requires 'nISM','nCore','rCloud','rCore','dens_profile','densPL_alpha'. A: powerLawSphere.py:214 defaults mu=1.4 and nISM=1.0 (both code-unit mismatches, see S2-R-11/S2-R-20) and the function is not called anywhere in the slice. C (S2-C-27) adds the physics requirement: the returned triple must simultaneously satisfy M(rCloud)=mCloud, rCore = f*rCloud and nEdge >= nISM - which fails for a diffuse cloud (nCore = 50, alpha = -2, f = 0.1 gives nEdge = 0.5 cm^-3 < nISM).",
    "expected": "Either return TRINITY key names and validate the nEdge constraint (widening rCore per S2-R-09's formula if needed), or stop advertising it as the recommended setup path.",
    "failure_scenario": "A test that follows the docstring passes the returned dict to get_density_profile and hits KeyError on 'dens_profile'/'densPL_alpha' - or, if .get() with a default is used, silently evaluates a homogeneous profile with alpha defaulting to 0.",
    "repro": "get_density_profile(1.0, compute_consistent_params(M_cloud=1e5, nCore=..., alpha=-2)); and compute_consistent_params(1e5, 50.0, -2.0, nISM=1.0) - check the returned nEdge against nISM.",
    "confidence": "medium",
    "lenses": ["A", "B", "C"],
    "divergence": "BC",
    "status": "corroborated",
    "source_ids": ["S2-B-12", "S2-C-27", "S2-A-11"]
  },
  {
    "id": "S2-R-17",
    "file": "trinity/cloud_properties/bonnorEbertSphere.py",
    "line": 357,
    "class": "divergence",
    "severity": "S4",
    "claim": "Two problems at the stability boundary. (a) Prose uses rejection language ('Omega must be < 14.04 for stability') for something the code only warns about - and C confirms warning is the correct behaviour, so the prose is stale. (b) The warning and the is_stable flag use inconsistent comparisons: at Omega == 14.04 exactly, is_stable is False but no warning is logged.",
    "evidence": "A: line 357 'if Omega > OMEGA_CRITICAL' logs the warning; line 363 'is_stable = Omega < OMEGA_CRITICAL'; supercritical spheres are accepted and run. B: bonnorEbertSphere.py:307 'must be < 14.04 for stability' with validate=True by default, vs validate_gmc.py:518 'Stability warning'. C: 'flag Omega > 14.042 as gravitationally unstable' - i.e. warn, do not reject - and notes the shipped default Omega = 14.1 is already marginally supercritical (xi_out = 6.4617), which C calls a documented-flag item rather than a bug.",
    "expected": "One boundary convention for both comparisons, and prose that says 'warned, not rejected'. Add an explicit note that the default Omega = 14.1 is marginally supercritical by design.",
    "failure_scenario": "Cosmetic: a run at exactly the critical Omega is reported inconsistently between the log and the validation result; a user reading 'must be < 14.04' expects a hard error and gets a warning.",
    "repro": "create_BE_sphere(..., Omega=14.04) and inspect is_stable plus the captured log; then Omega=20.0 through both create_BE_sphere and validate_gmc_params.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "BC",
    "status": "corroborated",
    "source_ids": ["S2-A-15", "S2-B-14", "S2-C-11"]
  },
  {
    "id": "S2-R-18",
    "file": "trinity/cloud_properties/bonnorEbertSphere.py",
    "line": 3,
    "class": "citation",
    "severity": "S4",
    "claim": "The BE critical constants are numerically correct, but the module docstring mislabels them: it lists 'm_crit ~ 1.182 (critical dimensionless mass)' immediately alongside 'mass formula m(xi) = xi^2 du/dxi', for which the critical value is 15.70; 1.182 belongs to Bonnor's different convention m_B = m/sqrt(4 pi Omega).",
    "evidence": "A re-solved the Lane-Emden equation and got xi(Omega=14.04) = 6.4504, m(xi_crit) = 15.703, implied Bonnor coefficient 1.1822 - confirming all four stored constants are numerically right, and that three of them (XI_CRITICAL, M_DIM_CRITICAL, M_BONNOR_CRITICAL) are never referenced anywhere. C computed the same numbers independently (6.4507514 / 14.042032 / 15.704374 / 1.1822266) and supplied the exact relation M_BONNOR_CRITICAL = M_DIM_CRITICAL/sqrt(4 pi * OMEGA_CRITICAL) as a one-line self-test (invariant I-9). B found the docstring/comment-block contradiction and checked the conversion by hand (0.2821 * 0.2669 * 15.70 = 1.182), concluding the comment block is right and the module docstring mislabels.",
    "expected": "The module docstring should read m_crit ~ 15.70 for the m = xi^2 du/dxi convention it says is used. Add the I-9 assertion as a test against the module's own solver. NOTE: neither A nor C had literature access, so the numeric VALUES are corroborated by two independent integrations but the literature NAMES attached to them remain unconfirmed - that, and only that, is pending.",
    "failure_scenario": "Low blast radius today because the three constants are dead code. If a future stability check compares an m computed as xi^2 du/dxi against the 1.182 constant, every sphere is misclassified (15.70 > 1.182 always).",
    "repro": "assert abs(M_BONNOR_CRITICAL - M_DIM_CRITICAL/sqrt(4*pi*OMEGA_CRITICAL)) < 1e-6 and abs(OMEGA_CRITICAL - exp(u(XI_CRITICAL))) < 1e-3, using the module's own solve_lane_emden.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "BC",
    "status": "corroborated",
    "source_ids": ["S2-B-03", "S2-C-10", "S2-C-11", "S2-C-12", "S2-A-16"]
  },
  {
    "id": "S2-R-19",
    "file": "trinity/cloud_properties/bonnorEbertSphere.py",
    "line": 91,
    "class": "regime",
    "severity": "S4",
    "claim": "XI_MAX = 20.0 caps the representable density contrast at Omega ~ 222, and that ceiling is documented nowhere - but the code fails loudly rather than clamping, so C's feared silent extrapolation does not occur.",
    "evidence": "A: 'target_rho_rhoc < rho_rhoc[-1] -> ValueError', and with XI_MAX = 20 the interpolator's last tabulated ratio is 1/222.3, capping Omega at 222.27. B (S2-B-15) noted no maximum supported Omega is documented anywhere even though the table depth sets one. C (S2-C-18) computed the Omega -> xi_out map (100 -> 14.2156, 1e3 -> 40.8176, 1e4 -> 139.4178) and required an out-of-range Omega to fail loudly, never clamp - which is what A observed.",
    "expected": "Document the ceiling alongside 'must be < 14.04', and make the error message name Omega rather than the interpolation table.",
    "failure_scenario": "A large-Omega request raises an out-of-range error whose message points at the density table rather than at Omega. No silent wrong answer.",
    "repro": "create_BE_sphere with Omega = 500 and inspect the exception message.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S2-B-15", "S2-C-18"]
  },
  {
    "id": "S2-R-20",
    "file": "trinity/cloud_properties/validate_gmc.py",
    "line": 186,
    "class": "units",
    "severity": "S4",
    "claim": "check_gmc_constraints defaults nISM=1.0, which in code units is 1 particle per cubic parsec (3.4e-56 cm^-3), so if the default is ever taken the nEdge >= nISM test is unconditionally satisfied for any realistic nEdge. The module's own usage example omits nISM and therefore relies on that default - and then still fails two checks.",
    "evidence": "A: validate_gmc.py:186 'nISM=1.0'; line 237 'if nEdge < nISM'; realistic nISM = 1 cm^-3 is 2.938e55 in these units. All in-slice callers pass nISM explicitly, so the default is latent. B: the documented example check_gmc_constraints(rCloud=150.0, nEdge=0.5, mCloud=1e5, M_computed=1.001e5) omits nISM; with the 1.0 default, 0.5 < 1.0 fires the edge-density error, and |1.001e5-1e5|/1e5 = 1.0000000000001455e-3 > 0.001 fires the mass error (A confirms the comparison is strict '>' with an inclusive pass at exactly the tolerance).",
    "expected": "No default (or 1.0*ndens_cgs2au), and a usage example that runs clean.",
    "failure_scenario": "A direct caller omits nISM and the edge-density sanity check silently becomes a no-op; a user copying the documented example sees two errors for a cloud the author intended as passing.",
    "repro": "check_gmc_constraints(10.0, 1e-40, 1e6, 1e6) -> no 'Edge density' error; then run the validate_gmc.py:3 example verbatim and print the returned errors.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S2-A-12", "S2-B-17"]
  },
  {
    "id": "S2-R-21",
    "file": "trinity/cloud_properties/density_profile.py",
    "line": 126,
    "class": "numerical",
    "severity": "S4",
    "claim": "Two overstated smoothing claims. (a) 'the rhs is C^infty everywhere' is false: the profile has a slope kink at rCore for alpha != 0, and the BE branch goes through a cubic interpolant (C^2 at best). (b) 'mass conservation holds to O(SMOOTH_FRAC^2)' does not describe M(rCloud), where the error is linear in SMOOTH_FRAC.",
    "evidence": "A: the blend is a single global tanh applied to both profiles (so B's fear that densBE is unsmoothed is unfounded), but the core/envelope join at rCore is only C^0 in slope, and f_rho_rhoc is a cubic interp1d. A's measured M(rCloud) discrepancies (1.03% at alpha=0, 0.37% at alpha=-2, 0.57% for BE) are O(SMOOTH_FRAC = 0.01), not O(1e-4). B (S2-B-27) predicted exactly this degradation and named the condition under which O(w^2) would hold. The O(w^2) claim is defensible for the mass integrated across the whole band (the tanh is antisymmetric in linear r about rCloud) - just not at rCloud.",
    "expected": "Scope the smoothness claim ('C^infty at the cloud edge', not 'everywhere') and state the mass-conservation bound for the quantity that is actually used, M(rCloud).",
    "failure_scenario": "Documentation only - but it is the claim that makes S2-R-01 look already-bounded, so it actively conceals a 10x-over-tolerance error.",
    "repro": "Finite-difference n(r) and its first three derivatives across rCore and across rCloud*(1 +/- 2*SMOOTH_FRAC) for both dens_profile values; then halve SMOOTH_FRAC and confirm the M(rCloud) error halves rather than quartering.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S2-B-09", "S2-B-27", "S2-A-01"]
  },
  {
    "id": "S2-R-22",
    "file": "trinity/cloud_properties/density_profile.py",
    "line": 143,
    "class": "numerical",
    "severity": "S4",
    "claim": "For alpha < 0 the power-law expression is evaluated over the whole radius array - including r = 0 - before the core mask is applied, producing inf and a numpy divide-by-zero RuntimeWarning on every call that includes the origin. The returned value is correct; only the warning is spurious.",
    "evidence": "A: line 143 computes n_inside = nCore*(r_arr/rCore)**alpha over the whole array; line 146 only afterwards overwrites n_inside[r_arr <= rCore] = nCore. With r_arr[0] = 0 and alpha < 0, (0/rCore)**alpha = inf.",
    "expected": "Evaluate the power only where r > rCore (np.where on the mask, or clip r to rCore first).",
    "failure_scenario": "Warning noise in every profile plot or grid evaluation that starts at r = 0, which masks genuine numerical warnings from other modules.",
    "repro": "get_density_profile(np.linspace(0, rCloud, 100), params) with densPL_alpha = -2.",
    "confidence": "high",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S2-A-17"]
  },
  {
    "id": "S2-R-23",
    "file": "trinity/cloud_properties/validate_gmc.py",
    "line": 287,
    "class": "divergence",
    "severity": "S4",
    "claim": "rCore=None with alpha != 0 is advertised as supported by the solver (rCore_fraction = 0.1 fallback) and rejected by the validator ('required for densPL with alpha != 0', and _validate_powerlaw raises).",
    "evidence": "B: validate_gmc.py:287 'rCore ... (required for densPL with alpha != 0)' vs powerLawSphere.py:78 'rCore : optional. If None, uses rCore_fraction (default 0.1)'. A confirms the validator side actually raises: '_validate_powerlaw:412 already raises in that case' (which is also why the potential TypeError at validate_gmc.py:592-593 is unreachable). B additionally notes the two branches are not equivalent - the fixed-rCore branch has a mass ceiling for alpha < -3 that the fractional branch does not - so which one runs is physically observable.",
    "expected": "One story about whether rCore=None is supported for densPL with alpha != 0.",
    "failure_scenario": "Validation rejects a parameter set the underlying solver would have handled via the 0.1 fraction - a capability advertised in one module and denied in another.",
    "repro": "validate_gmc_params(..., dens_profile='densPL', alpha=-2, rCore=None) and compare with compute_rCloud_powerlaw(..., rCore=None).",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S2-B-23", "S2-A-05"]
  },
  {
    "id": "S2-R-24",
    "file": "trinity/cloud_properties/powerLawSphere.py",
    "line": 157,
    "class": "divergence",
    "severity": "S4",
    "claim": "For alpha < -3 the fixed-rCore inversion's rejection message is wrong: rhs <= 0 there means the requested mass exceeds the sphere's ASYMPTOTIC TOTAL mass, not that 'the uniform core alone exceeds the cloud mass budget'. DEMOTED from B's S3: the guard itself is correct and complete, so the NaN B feared cannot occur.",
    "evidence": "A: for 3+alpha > 0 the rhs <= 0 test is the correct and complete condition (M below the core mass); for 3+alpha < 0 the forward map is decreasing in rCloud, so rhs <= 0 encodes the asymptotic-mass ceiling instead - and the result is self-checked at powerLawSphere.py:166-173 (rel_err > 1e-6 -> RuntimeError), so a bad root cannot escape. B (S2-B-06) predicted 'negative base ** negative power -> NaN', which A's account rules out. C's validity table independently identifies the same ceiling: for alpha <= -3 the envelope mass converges, so a requested M above that ceiling has no root.",
    "expected": "Regime-aware message text, or an explicit statement that alpha < -3 is unsupported. No code-path change needed.",
    "failure_scenario": "A user hitting the alpha < -3 mass ceiling is told to reduce rCore, which does not help.",
    "repro": "compute_rCloud_powerlaw(M_cloud large, nCore, alpha=-3.5, rCore=1.0) and read the raised message.",
    "confidence": "medium",
    "lenses": ["A", "B", "C"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S2-B-06", "S2-A-04"]
  },
  {
    "id": "S2-R-25",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 331,
    "class": "citation",
    "severity": "S4",
    "claim": "'Rahner+ 2018, Eq 25' is cited three times (for the enclosed-mass formula and for its inversion) with no volume or page, while the only fully-specified Rahner reference in the slice is 'Rahner et al. (2017): MNRAS 470, 4453'.",
    "evidence": "B: mass_profile.py:331, powerLawSphere.py:78 and :149 all cite 'Rahner+ 2018 Eq 25'; bonnorEbertSphere.py:3 lists 'Rahner et al. (2017): MNRAS 470, 4453'. Two different Rahner years for related WARPFIELD physics inside one package. MITIGATION: the formula being cited is independently verified correct by A (integration, 4.7e-12) and C (quadrature, 4e-16), so the exposure is bibliographic only.",
    "expected": "A complete reference for the 2018 citation, and confirmation that Eq 25 of that paper is this expression (including whether their rho_c is a mass or a number density).",
    "failure_scenario": "A wrong equation number or paper propagates into a methods section; a reader cannot verify the prefactor or the (3+alpha) convention.",
    "repro": "Check Eq 25 of the intended Rahner et al. paper against the formula as written.",
    "confidence": "medium",
    "lenses": ["B"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S2-B-21"]
  },
  {
    "id": "S2-R-26",
    "file": "trinity/cloud_properties/validate_gmc.py",
    "line": 631,
    "class": "other",
    "severity": "S4",
    "claim": "The parameter-suggestion machinery ranks candidates with a distance metric that mixes log10 distances (mCloud, nCore) with a linear fractional distance (rCore), so the ranking is not on a common scale; and no lens found evidence that suggestions are round-tripped through the validator before being offered.",
    "evidence": "A: validate_gmc.py:631-634 mixes the two scales; the searches are pure grid scans over +/- factor arrays with n_combos correctly subtracting the identity combo; search_range=0.5 (lines 551, 643) is declared and never referenced. C (S2-C-24) states the one hard requirement: every suggestion returned must itself pass the validator when fed back in - a self-consistency test costing one loop.",
    "expected": "A single distance scale, and a round-trip validation of each suggestion before it is displayed.",
    "failure_scenario": "The user is handed 'closest valid' parameters that are neither closest nor guaranteed valid on the next run.",
    "repro": "Round-trip every suggestion through validate_gmc_params for a deliberately failing input; and check whether reordering by a uniform log-distance changes the top-3.",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S2-C-24", "S2-A-16"]
  },
  {
    "id": "S2-R-27",
    "file": "trinity/cloud_properties/bonnorEbertSphere.py",
    "line": 79,
    "class": "deadcode",
    "severity": "S4",
    "claim": "Pre-existing dead code, flagging only (project rule: do not delete): XI_CRITICAL, M_DIM_CRITICAL and M_BONNOR_CRITICAL are defined and never used (only OMEGA_CRITICAL is); validate_gmc.py declares search_range=0.5 twice and never references it; mass_profile.py imports compute_rCloud_homogeneous and compute_rCloud_powerlaw without calling them; bonnorEbertSphere imports scipy.optimize and binds M_H_CGS and MYR_TO_S unused.",
    "evidence": "A enumerated all of these; ruff is configured for F821/F811/F823/E9 only, so F401 does not catch the unused imports. The three unused constants are numerically correct (see S2-R-18).",
    "expected": "Flag only, per the project rule on pre-existing dead code.",
    "failure_scenario": "",
    "repro": "grep -rn 'XI_CRITICAL\\|M_DIM_CRITICAL\\|M_BONNOR_CRITICAL\\|search_range' trinity/cloud_properties/",
    "confidence": "high",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S2-A-16"]
  },
  {
    "id": "S2-R-28",
    "file": "trinity/cloud_properties/density_profile.py",
    "line": 45,
    "class": "other",
    "severity": "S4",
    "claim": "The scalar-in/scalar-out contract is documented as returning a builtin float ('<class \\'float\\'>' in two doctests) but no lens verified the actual return type; np.float64 and 0-d arrays are the classic _is_scalar blind spots.",
    "evidence": "B recorded the doctest claims at density_profile.py:56 and mass_profile.py:137 plus the helper docstrings 'Convert result back to scalar if input was scalar'. C (S2-C-21, invariant I-14) states the full contract - scalar in => scalar out, array in => same-shape out, f(array)[i] == f(array[i]) - and notes the _is_scalar/_to_array/_to_output trio is duplicated verbatim in density_profile.py and mass_profile.py. Neither lens executed the check; A did not examine return types.",
    "expected": "assert type(get_density_profile(0.5, params)) is float, plus the elementwise consistency check across [1.0, np.float64(1.0), np.array(1.0), 1].",
    "failure_scenario": "Metadata/JSON writers relying on the builtin-float contract get np.float64 and either fail to serialise or emit a different textual representation - relevant to the project's byte-identical dictionary.jsonl equivalence gate.",
    "repro": "For x in [1.0, np.float64(1.0), np.array(1.0), 1]: compare type and value of get_density_profile(x, params) against get_density_profile(np.array([1.0]), params)[0].",
    "confidence": "low",
    "lenses": ["B", "C"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S2-B-18", "S2-C-21"]
  },
  {
    "id": "S2-R-29",
    "file": "trinity/cloud_properties/initial_profile.py",
    "line": 58,
    "class": "numerical",
    "severity": "S4",
    "claim": "The reconstructed (r, density, mass) arrays may not resolve rCore: with a small rCore and a 10-100 pc cloud the core spans ~1e-4 of the domain, so a linear grid of even 1e4 points puts ~1 point inside it.",
    "evidence": "C only, reasoning from the signature plus the spec's rCore default. C's expected checks: log-spaced radii or an explicitly refined core region; M[0] ~= (4pi/3) rho_core r[0]^3; M[-1] == mCloud to 1e-10; monotone M and non-increasing density. Neither A nor B examined the grid construction, so the grid type is unverified. Note CLAUDE.md's convention is rCore ~ 1 pc, which is far less severe than the 0.01 pc default C assumes.",
    "expected": "Log-spaced (or core-refined) radii, and M[-1] == mCloud to 1e-10.",
    "failure_scenario": "An unresolved core makes any grid-based enclosed mass miss the core contribution and biases every downstream interpolation of M(r) near the centre - exactly where the shell starts.",
    "repro": "build_initial_cloud_profile with dens_profile='densPL', alpha=-2, rCore=0.01, rCloud=30; count grid points with r < rCore and compare M[-1] to mCloud.",
    "confidence": "low",
    "lenses": ["C"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S2-C-25"]
  },
  {
    "id": "S2-R-30",
    "file": "trinity/cloud_properties/bonnorEbertSphere.py",
    "line": 64,
    "class": "units",
    "severity": "S4",
    "claim": "bonnorEbertSphere carries its own cgs constants block (G_CGS, K_B_CGS, M_H_CGS, MSUN_TO_G, PC_TO_CM, MYR_TO_S) duplicating trinity/_functions/unit_conversions.py; the two must agree and no lens diffed them.",
    "evidence": "A lists the module's cgs constants and their use, and confirms the dimensional chain is self-consistent within the module (it read unit_conversions.py only for the unit convention and constant values, not for a diff). C (S2-C-30) supplies the expected values and flags the 0.05% atomic-H vs proton-mass and 0.03% Msun-convention spreads. A's inferred rho/n product (0.034618) vs C's derived (0.0346101) differ by 0.02%, consistent with exactly this kind of convention spread.",
    "expected": "One source of truth, or an assertion that the two blocks agree.",
    "failure_scenario": "Independent constant blocks that drift apart make bit-identical equivalence tests fail for reasons unrelated to the change under test.",
    "repro": "Diff the constants in bonnorEbertSphere.py against trinity/_functions/unit_conversions.py.",
    "confidence": "low",
    "lenses": ["A", "C"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S2-C-30"]
  },
  {
    "id": "S2-R-31",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 608,
    "class": "silent-failure",
    "severity": "S4",
    "claim": "In the alpha = 0 branch of compute_minimum_rCore, nCore > nISM is stated as a premise ('Homogeneous: nEdge = nCore, always valid if nCore > nISM') rather than evaluated, while the function's documented return includes 'is_valid : Whether nEdge >= nISM'.",
    "evidence": "B only, from the comment text at mass_profile.py:608 against the documented return contract at :567. A confirms an explicit alpha == 0 branch exists (which refutes C's separate T-09/S2-C-07 division-by-zero concern) but did not report whether that branch evaluates the nEdge test.",
    "expected": "The alpha = 0 branch should evaluate nEdge >= nISM rather than assuming it.",
    "failure_scenario": "alpha=0 with nCore <= nISM returns is_valid=True on an unchecked premise, bypassing validate_gmc constraint #2 for homogeneous clouds.",
    "repro": "compute_minimum_rCore(nCore=nISM/2, nISM=nISM, rCloud=20.0, alpha=0.0) and inspect is_valid.",
    "confidence": "low",
    "lenses": ["B"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S2-B-26"]
  }
]
```
