# Conduction-zone luminosity convergence audit

Companion to `bubble-integrator-robustness.md`. That document covers the
*crash* (LSODA "illegal input" on the over-refined grid). This one quantifies
the *accuracy* question that surfaced while rejecting the grid-de-refinement
fix (Option 1): **how converged is the production conduction-zone luminosity,
and what is the converged value?**

It is the sign-off package for the `solve_ivp(dense_output=True)` integrator
switch (Commit 2/3 of the bubble-solver fix): it establishes the target
numbers *before* any runtime behavior changes. **No runtime behavior is
changed by this document or its tool.**

## TL;DR

- **The production `odeint`@20k conduction luminosity is well converged.** Over
  12 sampled Phase-1a states of the quickstart scenario, the production
  `bubble_L2Conduction` is within **±0.17 %** of the converged value at every
  state except one: the **thin early-bubble state** (`R1 ≈ r2Prime`), where it
  is **−0.89 %** low. Mean bias **−0.07 %**, max **0.89 %**.
- **The 12 % / 6.5 % shifts reported earlier were Option-1 artifacts.** They
  came from *reducing* the Step-2 grid count to 500 (under-resolving the
  trapezoid), not from the production 20k resolution. They proved grid
  de-refinement is non-viable; they are **not** evidence that 20k is wrong.
- **`solve_ivp(dense_output=True)` reaches the converged value efficiently and
  never fails.** Across all 12 states × 3 `rtol`s: **0 integration failures**,
  typical **~1000 internal steps** (vs `odeint` interpolating ~hundreds of
  steps onto ~60k near-duplicate output radii and intermittently crashing).
- The converged integral is **`rtol`-independent** (changes < 0.001 % between
  `rtol` 1e-6 and 1e-10); accuracy is set by the cheap output sampling, not the
  integrator. This is the "efficiency without brute force, but properly
  resolved" property: integrate once cheaply, refine the quadrature for free.

## Method

`tools/bubble_conduction_convergence.py`. Single-threaded (BLAS pinned), so the
numbers are deterministic. For each sampled Phase-1a state of a driven run:

1. **Production reference** — `bubble_L2Conduction` from the unmodified
   `get_bubbleproperties_pure` (`odeint` on the ~60k legacy grid).
2. **Converged ground truth** — re-integrate the structure with
   `solve_ivp(method='LSODA', dense_output=True, rtol=1e-10)`, then compute the
   conduction integral by sampling the *continuous* solution `sol.sol(r)` at
   `K ∈ {500, 2k, 10k, 50k, 200k}` points and taking the value at `K=200k`
   (converged to < 0.005 % between the last two K). The conduction integral
   replicates `_bubble_luminosity_legacy`'s non-CIE block exactly, including the
   production `T < 10**5.5` mask.
3. **Bias** = `(production − converged) / converged`.
4. **Mechanism/efficiency** — `solve_ivp` success and internal step count at
   `rtol ∈ {1e-6, 1e-8, 1e-10}`.

Scenario: the quickstart smoke param (`mCloud 1e5`, `sfe 0.3`, `stop_t 1e-4`),
auditing every 8th Phase-1a state (12 states total).

Reproduce:

```
python tools/bubble_conduction_convergence.py --stride 8 --max-states 12
```

## Results (quickstart scenario, 12 states)

| state | R1 (pc) | r2Prime (pc) | prod L2Conduction | converged | bias % | solve_ivp steps @1e-8 |
|------:|--------:|-------------:|------------------:|----------:|-------:|----------------------:|
| 0  | 0.00110 | 0.00127 | 1.01405e+06 | 1.02318e+06 | **−0.89** | 950 |
| 8  | 0.05494 | 0.12298 | 6.24920e+06 | 6.25412e+06 | −0.08 | 938 |
| 16 | 0.05370 | 0.14902 | 1.11132e+07 | 1.11089e+07 | +0.04 | 999 |
| 24 | 0.05399 | 0.16817 | 1.48221e+07 | 1.48109e+07 | +0.08 | 962 |
| 32 | 0.05504 | 0.18450 | 1.76962e+07 | 1.76752e+07 | +0.12 | 1017 |
| 40 | 0.05650 | 0.19928 | 1.99629e+07 | 1.99344e+07 | +0.14 | 1023 |
| 48 | 0.05817 | 0.21305 | 2.18102e+07 | 2.17742e+07 | +0.17 | 1040 |
| 56 | 0.05994 | 0.22611 | 2.32727e+07 | 2.33052e+07 | −0.14 | 1017 |
| 64 | 0.06177 | 0.23859 | 2.45802e+07 | 2.46077e+07 | −0.11 | 995 |
| 72 | 0.06363 | 0.25061 | 2.57173e+07 | 2.57408e+07 | −0.09 | 1025 |
| 80 | 0.06548 | 0.26221 | 2.67303e+07 | 2.67487e+07 | −0.07 | 1034 |
| 88 | 0.06731 | 0.27346 | 2.76511e+07 | 2.76618e+07 | −0.04 | 1024 |

**Summary:** mean bias −0.07 %, max |bias| 0.89 % over 12 states.
**solve_ivp failures across all states × rtols: 0.** Typical step count ~1017.

## Reading the result

- **The outlier is the thin early bubble (state 0).** There `R1 ≈ r2Prime`
  (shell thickness ~1.7e-4 pc), the conduction zone is most geometrically
  compressed, and the production trapezoid is most under-resolved → −0.89 %.
  This is the *same* thin-shell geometry that makes the legacy grid produce
  near-duplicate radii and crash LSODA (see `bubble-integrator-robustness.md`
  §I.6). The two pathologies share a root: a thin conduction zone forced onto a
  fixed logspace grid.
- **Everywhere else the production value is fine** (|bias| ≤ 0.17 %). So the
  conduction luminosity numbers are not meaningfully wrong at production
  resolution; the priority remains the crash.
- **Impact on `LTotal`:** `L_conduction` is one additive component of
  `bubble_LTotal`, so its ≤0.9 % bias *upper-bounds* its contribution to the
  `LTotal` error (smaller in fraction, since `L_conduction < LTotal`). The
  converged `LTotal` itself is delivered by the Commit 3 end-to-end output-diff;
  this audit bounds the conduction term going in.

## Implication for the fix

The `solve_ivp(dense_output=True)` switch:

1. **removes the crash** (the integrator never sees near-duplicate output
   radii; 0 failures across all states here), and
2. **tightens the residual conduction bias to the converged value** as a
   byproduct, without brute-forcing the grid — the integrator runs ~1000 steps
   and accuracy comes from cheap dense-output sampling.

Because the production bias is already small (≤0.9 %, mostly ≤0.2 %), the
expected success-path output change at Commit 3 is **small and well-bounded** —
a tractable, explicitly-documented physics correction for model-author sign-off,
not a silent shift.

## Scope / caveats

- One scenario (quickstart `mCloud 1e5 / sfe 0.3`), 12 sampled states. Before
  final sign-off on Commit 3, re-run on a second case (e.g. a low-SFE /
  high-density run, where the conduction zone and switch indices differ) to
  confirm the bias stays bounded. The tool takes `--param` / `--param-text`.
- The audit measures `L_conduction` only. The full `LTotal` (CIE + conduction +
  intermediate) convergence is the Commit 3 output-diff deliverable.
- `solve_ivp` here uses `method='LSODA'` to match the production stiff solver;
  Commit 2 will confirm the success-path delta vs `odeint` on non-failing runs.
