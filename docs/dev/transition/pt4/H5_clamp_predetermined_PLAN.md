# H5 — is legacy's transition "predetermined" by the β/δ clamp? (plan)

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
> a committed artifact (a CSV/table under `docs/dev/data/`, or a force-added
> harness/figure in the relevant `docs/dev/<workstream>/` folder) — never left in
> `/tmp` or an untracked `outputs/`. A future visit must be able to reproduce or
> compare against the numbers **without re-running**; record the exact config +
> command that produced each artifact.
>
> 🔗 **Cross-check the sibling docs — keep the workstream self-consistent.** This file is one of
> several living docs for its workstream (its `PLAN.md`, `FINDINGS.md`, `runs/README.md`, `NOTE_PATCHES.md`,
> and any other notes in the same folder). They drift out of sync *with each other* as fast as they drift
> from the code. Any agent or person editing one MUST, as part of the visit, circle back through the
> siblings and reconcile: if a number, status, claim, or line reference here contradicts a sibling — or a
> sibling has gone stale — fix it (or flag it, dated) so no two docs in the workstream disagree. Never
> update one in isolation.

**Date:** 2026-06-22. **Status:** plan (not yet run). No production change (monkeypatch + offline only).

## The hypothesis (maintainer; tested skeptically, NOT assumed true)

Legacy clamps `(β,δ)` to the box `[0,1]×[−1,0]` (`get_betadelta.py:41–44`, enforced by `np.clip`
:1044–45 and the L-BFGS-B `bounds` :1053). H5 claims: this clamp **soft-locks** `(β,δ)` on the box
boundary, which **holds `Lloss` artificially high**, so the cooling ratio `(Lgain−Lloss)/Lgain`
**inevitably declines below 0.05 → a "predetermined" transition** — i.e. legacy's "let cooling decide"
is really a clamp artifact, not physics.

## My assessment (evidence so far — to be confirmed/refined, not assumed)

- **Partially supported already.** pt4-H1 found legacy's crossing is a *"constrained edge-root artifact
  of the clamp"*: the true (hybr, unbounded) root is `β≈+3.5…+4.2`, clamped down to ≤1; that low β
  keeps `Lloss` high so the ratio falls and crosses, whereas hybr's free β collapses `Lloss` and the
  ratio recovers (never crosses). So the clamp **does** bias toward crossing.
- **But "inevitable / always" is likely too strong.** `large_diffuse_lowsfe` does **not** cross under
  legacy (min ratio 0.514) — a standing counter-example. So the clamp is plausibly *necessary but not
  sufficient*: it biases toward crossing, the config's cooling still matters. Expected refined verdict:
  **clamp-contingent, not "predetermined."**
- This is **falsifiable** (see gates). If β is *interior* (not on the boundary) at the crossing, or if
  un-clamping does *not* remove the crossing, H5 is wrong.

## Test design — two parts

### Part A — soft-lock characterization (OFFLINE, committed data, no re-run)
From `../cleanroom/data/c0_<cfg>_legacy.csv` and `_h0.csv` (cols `cool_beta, cool_delta, bubble_Lgain,
bubble_Lloss, t_now`). Per config:
- **Boundary-pin fraction:** fraction of segments (and of the segments leading to the 0.05 crossing)
  where legacy `(β,δ)` sits on the box boundary (`β≈1` or `β≈0` or `δ≈−1` or `δ≈0`, within ε). High
  fraction near the crossing ⇒ soft-lock.
- **β at the crossing:** is legacy β pinned at the upper bound (1.0) when the ratio crosses?
- **Counterfactual (hybr):** the unbounded β at the same epochs (how far above 1 the true root wants to
  go — the "clamp gap"). Large gap ⇒ the clamp binds hard.
- **Trend table** across all 6 configs: `boundary_pin_frac`, `β_at_cross`, `hybr_β_max`, `crosses?`,
  `cross_t`, `ratio_min`. Expected pattern: crossing configs are boundary-pinned with a large hybr-β
  gap; `large_diffuse` either less pinned or crosses-not.

### Part B — clamp-width sweep (SIM, monkeypatch, the DECISIVE causal test)
Monkeypatch the legacy clamp's **upper β bound** `BETA_MAX` (and report δ) to progressively widen it,
then run `betadelta_solver=legacy` on each config and watch the 0.05 crossing:

`BETA_MAX ∈ {1.0 (default legacy), 2.0, 4.0, 10.0}`  + the **∞ reference = hybr** (already in
`c0_*_h0.csv`).

**Causal logic:** as the clamp loosens, β can approach its true (hybr) value, `Lloss` collapses, the
ratio recovers, and the crossing should **push later and then vanish**. If it does → the crossing is
*caused* by the clamp (H5's mechanism confirmed). If the crossing **persists** at wide bounds → the
transition is genuine cooling, **H5 refuted**.

Harness: a `h5_variants.py` monkeypatch of `get_betadelta.BETA_MAX` (+ optional `DELTA_MIN`), reusing
the failed-large-clouds/`h3_run_variant.py` driver pattern (V0 = run as-is with the patched module
constant). One sim/process, `OMP_NUM_THREADS=1`, `timeout`-bounded, `stop_t` per config large enough to
reach either the crossing or blowout. **6 configs × 4 widths = 24 cells**, parallel.

## The trend to look for (and the falsification)

- **Headline trend (Part B):** per config, *crossing time vs BETA_MAX* — monotonically later, then `None`
  (no cross) as the clamp widens toward hybr. A clean monotone family across all 6 = H5's mechanism.
- **Supporting (Part A):** crossing configs are boundary-pinned (soft-lock) with a large hybr-β gap;
  `large_diffuse` is the natural control (no cross even clamped).
- **Falsify H5 if:** (a) legacy β is interior (not on the boundary) at the crossing; or (b) the crossing
  survives at wide `BETA_MAX`; or (c) widening the clamp does **not** monotonically delay the crossing.
- **Likely refinement:** "clamp-**biased / contingent**" (the clamp holds Lloss high and *enables* the
  crossing) rather than "**predetermined / inevitable**" (since `large_diffuse` doesn't cross). The
  experiment will say which, with numbers.

## Subagent strategy

1. **Part A — 1 subagent (or inline), offline, fast:** mine the committed `c0_*_{legacy,h0}.csv`,
   compute the boundary-pin/trend table + the legacy-vs-hybr β figure. No sims. Minutes.
2. **Part B — matrix subagent(s) in worktrees, parallel:** build `h5_variants.py` (monkeypatch
   `BETA_MAX`), run the 24 cells across the 4 cores (≤4 concurrent), collect `(config, BETA_MAX,
   cross_t, ratio_min, reached_phase)` rows + per-cell ratio/β trajectories. Persist CSVs.
3. **Synthesis (parent):** plots (crossing-time-vs-clamp-width family; β legacy-vs-hybr; pin-fraction)
   + a findings doc with the verdict. Cross-validate Part B's `BETA_MAX=1` against the committed legacy
   data and `BETA_MAX→∞` against `c0_*_h0.csv` (hybr) — both must match.

## Gates
- **No production change** — `BETA_MAX` patched only in the harness (module monkeypatch), exactly like
  the failed-large-clouds variants. Production `get_betadelta.py` untouched.
- **Consistency:** `BETA_MAX=1.0` reproduces committed legacy; `BETA_MAX→∞` reproduces hybr.
- **Persist** every CSV/figure under `docs/dev/transition/pt4/h5clamp/`; record exact commands.

## Relationship to the rest of pt4
This is the *diagnosis-of-legacy* counterpart to the R1 work: R1 builds a defensible transition for the
correct (hybr) regime; H5 quantifies *why the legacy transition that R1 replaces was itself suspect*
(clamp-contingent). Together they justify not simply reverting to legacy.
