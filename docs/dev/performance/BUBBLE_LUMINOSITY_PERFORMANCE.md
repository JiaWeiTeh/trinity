# Bubble-luminosity performance — the complete arc

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

---

- **Status (2026-06-19):** 📘 **REFERENCE — the consolidated history of every performance/robustness change to `bubble_luminosity.py`.** All four eras have SHIPPED. This doc folds the (now-archived) F1 planning docs into one de-staled narrative; the live reference table is `F1_SUMMARY.md`, the illustrated version is `F1_REPORT.html`, and the granular F1 plan/patch are in `docs/dev/archive/bubble/`.
- **Scope:** `trinity/bubble_structure/bubble_luminosity.py` — the Weaver+77 bubble-structure solver. Via an `fsolve` over the mass-loss rate $\dot M$, it is the dominant per-step cost of a run.

## TL;DR — the life of the 60 000-point grid
The whole arc is the story of one number. The ~60k-point radius grid was **born load-bearing** (it was `odeint`'s integration request), became **vestigial** (a `solve_ivp` migration decoupled accuracy from sampling), and was finally **removed** from the hot residual (F1):

| era | change | effect | bit-exact? |
|---|---|---|---|
| **A** | `odeint` crash fix → `solve_ivp(dense_output)` migration | killed a 1-in-3 nondeterministic crash; **decoupled integration accuracy from output sampling** (the 60k becomes a mere output grid); residual-solve wall −10.4% | no (bounded: ≤0.28% dMdt, 0/67 fields >1%) |
| **B** | conduction zone: ~100-pt re-solve → sample dense solution at K=2000 | removed a fragile re-solve; bias tightened to the converged value (~7e-5 at K=2000, ~1 ms/call) | no (bounded physics correction, toward converged) |
| **C** | F2 "free wins" (`4a13075`) | `get_dudt` cooling cache **+23%/call**; dead-gravity integral removed; log ~340× smaller | **yes** (bit-identical) |
| **D** | **F1** (`24c6914`): drop the 60k resample in the dMdt residual | **~1.5×/call, ~2.3× full-run** on the degenerate config | no (full-run equivalent to ~6e-6, ≪ 0.3% gate) |

## Era A — the `odeint` crash, and the migration that demoted the 60k
**Problem.** `run.py` aborted nondeterministically (~1 in 3 identical re-runs) with a cryptic `MonotonicError`, surfacing a module away from the fault. The cause was **not** thread FP order (the original theory, superseded by a single-threaded fixed-seed repro): when LSODA bailed, `scipy.integrate.odeint` returned **uninitialised memory** for the un-integrated tail, and consuming that garbage made the whole solve nondeterministic.

**Origin of the 60k (the crux).** `odeint` integrates/interpolates onto *exactly the caller's output array*, so the ~60k near-duplicate radii (`_create_radius_grid`, packing 20k points into a ~1.6e-4 pc sliver, $dr\approx5\times10^{-9}$ pc) **were the integration request** — and asking LSODA to hit them inside a handful of real steps is what stressed its dense-output interpolation into the spikes/crash.

**Fix.** (1) Never consume a failed solve — check the success flag, return `_SOLVER_FAIL_RESIDUAL`/raise `BubbleSolverError`. (2) Migrate to `solve_ivp(dense_output=True)`, which **decouples integration accuracy (rtol/atol) from output sampling** — the integrator picks its own ~850–1000 adaptive steps and the grid is sampled from the *continuous* solution. **This demotes the 60k from a load-bearing integration grid to a vestigial output grid** — exactly what F1 later removes.

**Numbers.** Determinism repro: 794/786/792 `odeint` calls across identical runs → post-fix byte-identical. Migration diffs (98 snaps × 67 fields): dMdt shift ≤0.28%, 0/67 fields >1%; residual-solve wall **222.7 → 199.6 s (−10.4%)**. Commits `a245c29`(#659), `1eb7f4d`, `5f4f229`, `76921f7` (PR #666). Detail: `archive/bubble/integrator-robustness.md`.

**Lesson.** Detect & fail loud — never consume a failed solve. And *measure*: the non-determinism was memory garbage, not thread FP order; only a single-threaded fixed-seed repro proved it.

## Era B — conduction-zone convergence (the decoupling pays off)
**Problem → idea.** The conduction-zone luminosity used a fragile, under-resolved ~100-pt `odeint` re-solve (~0.9% low). Stop re-solving — *sample the already-computed dense solution* at a density chosen from a convergence study.

**Tested → result.** $K\in\{500,2\text{k},10\text{k},50\text{k},200\text{k}\}$ over 12 Phase-1a states. Convergence ~$1/K^2$; **K=2000 within ~7e-5** of $K\to\infty$ at the worst (thin-bubble) state, ~1 ms/call; integral `rtol`-independent (<0.001%); 0 failures.

**Outcome (`5f4f229`).** Deleted the re-solve + the well/under-resolved branch split; sample the dense solution at `_CONDUCTION_NPTS = 2000`. Wall neutral (+0.7%); bias tightened toward converged. Detail: `archive/bubble/conduction-convergence.md`.

**Lesson.** The same decoupling that fixed the crash bought accuracy *for free*: integrate once cheaply, refine the quadrature by sampling the continuous solution.

## Era C — F2 "free wins" (`4a13075`, bit-identical)
A hot-path audit harvested bit-identical cleanups around the bubble path (gated by a `git show HEAD` value-diff harness + the full non-stress suite, 535 passed):
- **F2.3+F2.4** — `get_dudt`: cache run-constant cooling cutoffs + move `Lambda_CIE` into its branch → **+23.1%/call (163.7→125.9 µs)**, exactly bit-identical (0 mismatches / 540 pts).
- **F2.2** — gravity outputs computed-then-discarded → disabled (`None` placeholders); removes a `simpson` + a 60k-divide per final structure solve.
- **F2.1** — `log_level` DEBUG→INFO: no wall win, but `trinity.log` ~340× smaller. (The "logging is the biggest free win" hypothesis was *retracted* after measuring.)
- **F2.5 dropped** — removing `pdotdot_total` is *not* bit-identical (it feeds the phase-1b A12 RHS). **F2.6 deferred.**

**Lesson.** Measure, don't guess — the audit retracted two of its own assumptions; the strict bit-identity bar kept "free" honest. (Follow-up `7f08e58` dropped a misleading `_legacy` suffix — sole production path all along.)

## Era D — F1: drop the vestigial 60k resample from the residual
**Problem.** `_get_velocity_residuals` (thousands of fsolve calls/run) still built the ~60k grid and evaluated the dense solution on all of it (`sol.sol(r_array)`) just to read $v$ at two endpoints and run the `min_T`/monotonic guards — a microbench put the resample at ~21 ms vs ~0.8 ms to integrate (~96% waste).

**Idea → options.** Integrate *once* on a coarse `t_eval` of $N$ points; drop the resample. A 6-variant matrix (`baseline` 60k, `M2000/M1000/M500/M200`, `Mnodes`) showed: every coarse option beats the baseline ~1.5× with speed ~flat across $N$, and accuracy is **npts-insensitive** (`M2000=M500=M200` to the digit, ~3e-6 ≪ the 0.3% gate). → **`_RESIDUAL_NPTS = 500`** = conservative robustness margin (guaranteed fixed-resolution guards vs `Mnodes`' variable adaptive nodes).

**Validation.** Per-call equivalence (≤3.1e-6, 6 configs) was *necessary but not sufficient*. The decider was **full-run equivalence**: original-60k vs F1-coarse, separate processes, compared at **matched simulation time**, on `mock_hybr` + three stiff edge cases (`simple_cluster`, low-ρ/hi-M/hi-sfe, hi-ρ/hi-M/lo-sfe) — worst R2/Eb **≈6e-6**, ~500× inside the gate. Plus 538 unit tests + opt-in stress (betadelta golden-match, bubble-solver 0-crash).

**Result.** ~1.5×/call; **~2.3× full-run** on the degenerate `simple_cluster`. The 60k was output over-resolution: `solve_ivp`'s adaptive stepping already resolves the stiff solution, so 500 points converge the guards to the same $\dot M$. Reference: `F1_SUMMARY.md`; illustrated: `F1_REPORT.html`; archived plan/patch: `archive/bubble/{RESAMPLE_PLAN,P3_PRODUCTION_PATCH}.md`.

## Methodology — how this workstream was run (for the next session)
Hard-won conventions worth reusing:

**Testing.**
- **A per-call / single-step equivalence is NECESSARY BUT NOT SUFFICIENT for a change to an iterative or integrated path.** A residual evaluated once at convergence cannot reveal that a coarse grid will mis-fire a guard on a *later, stiffer* step and compound. **Only a full-run equivalence test clears it** — and it must include the *stiffest / most degenerate* regimes (here: dense, weak-feedback edge cases), not just the easy config.
- **Bit-identical bar for "free" cleanups:** prove it, don't assert it — diff against `git show HEAD` (value-by-value) *and* require byte-identical `dictionary.jsonl` (sha256) on a smoke run. If a "never consumed" output turns out to feed an integrated RHS, it is not free (cf. F2.5).
- **Full-run A/B must use SEPARATE processes** — trinity carries module-level global state that leaks between two `start_expansion` calls in one process (this produced a *false* divergence once). Compare at **matched simulation time**, not final state (runs truncate at different `t`).
- **Measure, don't guess.** Retract a hypothesis the moment the data contradicts it (the "logging is the biggest win" and "thread FP order" theories were both wrong).

**Planning / implementation.**
- **De-risk before touching production:** build a P0 capture/replay harness and establish the baseline *first*; promote the production change only behind explicit gates (G0…). A config × method matrix beats a single-point check.
- **Persist diagnostics as committed artifacts** (CSVs in `data/`, figures + generators in the harness folder) so a future visit reproduces/compares *without re-running* the hours-long sims.
- **Surgical changes, bit-identical where claimed;** keep the diff traceable to the request.
- **Long runs:** drive them in the background with a heartbeat (keeps the container awake, surfaces failures) and an autosaver (commits completed artifacts every few minutes so a container reset can't lose them). Never commit while a file-swap experiment has production code checked out to a different version.
- **Docs discipline:** every plan/audit doc carries the 3 banners + a verified Status line; shipped work moves to `archive/`; update `README.md` + `DOC_STATUS.md` when docs move.
