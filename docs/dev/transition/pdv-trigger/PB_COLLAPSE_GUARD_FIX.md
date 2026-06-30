# Pb-fix plan — the energy-collapse reconciliation snapshot emits a garbage negative Pb

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/plan, not a maintained spec; the code moves faster than
> these notes (paths, line numbers, "what shipped" status drift). **Any agent or
> person reading this: treat it as unverified. Re-check each claim, snippet, and
> line reference against current source before relying on it.**
>
> 🔄 **Living plan — recheck and refine on every visit.** Re-verify the claims and
> line references above against current source; update anything that has drifted;
> rethink the strategy itself and note what changed and why (date it). Leave it
> better than you found it. **Keep all banner paragraphs at the top.**
>
> 💾 **Persist diagnostics — commit, don't re-run.** The negative-Pb evidence is the
> committed `runs/data/harvest_*` harvest (the `fail_repro` heavy run) — reproducible
> without re-running; record the exact command for any artifact added here.
>
> 🔗 **Cross-check the sibling docs — keep the workstream self-consistent.** Siblings:
> `INDEX.md`, `PLAN.md`, `FINDINGS.md`, `KMIX_PROTOTYPE.md` (§2 carries the original
> one-line diagnosis), `KMIX_IMPLEMENTATION_SPEC.md`. Reconcile any number/claim that
> disagrees; never update one in isolation.

---

## 0. Status, scope, and the guardrail this respects

**This is a PLAN only — no production code is changed by this document.** It is written under the
maintainer's standing guardrail (*nothing touches the production solver before it is tested*). The fix it
describes is a **one-line, correctness-preserving robustness change** to the energy-implicit phase; it is
queued here with its full test plan so it can be applied deliberately, not folded silently into the κ_mix
work. It is **independent** of the κ_mix effort — a pre-existing output-hygiene bug surfaced while harvesting
`Pb(t)` for the κ_mix prototype.

## 1. Symptom (what was observed)

The heavy-cloud verification run `fail_repro` (GMC 5×10⁹ M☉) writes a single **`Pb = −1.6×10¹⁸`** value in the
**last row** of its `dictionary.jsonl` / harvest. Every healthy run (`cal_diffuse/mid/compact/dense`, the
`f1edge_*`, `simple_cluster`) has **zero** negative `Pb`/`Eb` across 600+ rows. So the negative Pb is specific
to the energy-collapse stop fate, and appears exactly once, at the terminal row.

## 2. Root cause (traced line-by-line, 2026-06-30)

`Pb` is linear in `Eb`: `bubble_E2P(Eb, R2, R1, γ) = (γ−1)·Eb/V` (`bubble_structure/get_bubbleParams.py:236`,
called via `compute_R1_Pb` at `phase1b_energy_implicit/get_betadelta.py:329`). So `Eb < 0 ⟹ Pb < 0`, and the
extreme magnitude is a small-negative `Eb` divided by a tiny collapsing shell volume `V → 0`.

The control flow in `phase1b_energy_implicit/run_energy_implicit_phase.py`:

1. **Per-segment, healthy.** Each loop iteration computes `R1, Pb = compute_R1_Pb(R2, Eb, …)` at **line 865**
   from the *previous* segment's `Eb`, sets `params['Pb']` (line 868), and `save_snapshot()` at **line 946**
   records that consistent (positive) row *before* integrating the segment.
2. **Collapse detection.** After integrating, `Eb = float(sol.y[2, -1])` (**line 1055**); the guard
   `if not np.isfinite(Eb) or Eb <= 0:` (**line 1074**) sets `SimulationEndCode.ENERGY_COLLAPSED` (code **51**),
   `termination_reason = "energy_collapsed"`, and `break`s. **This guard is correct** — the stop fate, code,
   and reason all propagate properly. At this point `params['Pb']` still holds the *last healthy positive*
   value from step 1.
3. **The bug — the unconditional reconciliation snapshot.** After the loop, the *phase-boundary reconciliation*
   block (**lines 1269–1297**) runs **regardless of how the loop exited**. It recomputes
   `R1_f, Pb_f = compute_R1_Pb(R2, Eb, …)` (**line 1273**) using the **now-negative collapse `Eb`**, overwrites
   `params['Pb'] = Pb_f` (**line 1276**), and `save_snapshot()` (**line 1297**) writes that garbage `−1.6×10¹⁸`
   as the terminal row.

So the negative Pb is **not** a physics error and **not** from line 868 — it is the reconciliation block
recomputing derived properties from a post-collapse state the energy-driven model is explicitly invalid in
(the comment at lines 1069–1073 already says the energy model "would drive R1→R2 and divide-by-zero in
compute_R1_Pb" past collapse — the reconciliation block walks straight into exactly that regime).

## 3. Why it matters (and why it's low-severity but worth fixing)

- **Correctness of the stop fate: already right.** The run halts, with the right code (51) and reason.
  Downstream analysis already excludes `ENERGY_COLLAPSED` runs. Nothing physical is mis-integrated.
- **Output hygiene: wrong.** A `dictionary.jsonl` should never contain a `Pb = −1.6×10¹⁸`. It is a landmine for
  any consumer that does not special-case collapsed runs (plotting auto-scales blow up; an unguarded
  `log10(Pb)` or `mean(Pb)` poisons; the κ_mix harvest had to special-case it — see `KMIX_PROTOTYPE.md` §2).
- **Severity: low, fix: tiny.** One guard. The risk is entirely in *not regressing the healthy path*, which is
  why the test plan below is heavier than the fix.

## 4. The fix (smallest diff that is correct)

**Skip the reconciliation snapshot when the run ended by energy-collapse.** On that exit the last *healthy*
snapshot (step 1, line 946) is already the correct final physics row at the last valid `t`; recomputing
derived properties from the invalid post-collapse `Eb` produces only garbage. Guard the block:

```python
# phase1b_energy_implicit/run_energy_implicit_phase.py, at line 1269
if termination_reason != "energy_collapsed":
    try:
        feedback_final = get_current_sps_feedback(t_now, params)
        ...
        params.save_snapshot()
    except Exception as e:
        logger.warning(f"Phase-boundary reconciliation failed: {e}")
```

**Why this option over the alternatives:**

- ✅ **Chosen — skip reconciliation on collapse.** Minimal, intention-revealing, and *byte-identical for every
  non-collapsing run* (the guard is false only on the collapse path, which the healthy 8-config set never
  takes except `fail_repro`). The collapsed run keeps its last-valid row + the ENERGY_COLLAPSED code/reason.
- ❌ *Guard `Eb > 0` inside the block before `compute_R1_Pb`.* Also works, but leaves a half-reconciled
  snapshot (feedback updated, Pb/forces not) — a subtler inconsistency than simply not writing it.
- ❌ *Clamp `Pb` to NaN/last-valid in `bubble_E2P`.* Touches a shared physics primitive used everywhere — far
  larger blast radius, violates surgical-change. Rejected.
- ❌ *Reorder the line-1074 guard before the line-865 Pb compute.* Misdiagnosis — line 865/868 is not the
  source of the terminal garbage row (it holds the last *healthy* value); the reconciliation block at 1273 is.
  Reordering 1074 would not remove the bad row. **This corrects the earlier hypothesis recorded in
  `KMIX_PROTOTYPE.md` §2** (which guessed the line-1074-vs-865 ordering); the real culprit is lines 1269–1297.

## 5. Test plan (the fix is one line; the tests are the work)

All tests go in the `pytest` suite (project convention — no ad-hoc scripts). Three layers:

### 5.1 Unit — the collapse path short-circuits cleanly (the failing test, written first)

New `test/test_energy_collapse_snapshot.py`. Drive the energy-implicit phase to (or inject) an
`Eb ≤ 0` terminal state and assert on the resulting params/output:

- `params['SimulationEndCode'].value == SimulationEndCode.ENERGY_COLLAPSED.code` (== 51) — **propagation
  preserved**.
- `params['SimulationEndReason'].value` contains `"collapsed"` — **propagation preserved**.
- `params['Pb'].value > 0` (or is the last finite healthy value) — **no negative Pb survives**. *Before the
  fix this assertion fails* (`Pb ≈ −1.6×10¹⁸`); after, it passes. This is the bug-→-failing-test→-fix gate
  (rule 4).
- The recorded final row's `Pb` is finite and positive.

Prefer driving the real phase on the `fail_repro` config if it fits CI time; otherwise a minimal fixture that
calls the reconciliation path with `termination_reason = "energy_collapsed"` and a negative `Eb`, asserting no
`save_snapshot()` with negative Pb occurs (spy/monkeypatch on `save_snapshot`).

### 5.2 Regression — the healthy path is byte-identical (the "don't break anything" gate)

The fix must not perturb any non-collapsing run. Per CLAUDE.md rule 5, prove it on the solver edges:

- Run `param/simple_cluster.param` and `docs/dev/performance/f1edge_{lowdens,hidens}*.param` **before and
  after** the change, in **separate processes** (trinity leaks module-global state in-process), to the same
  `STOPPING_TIME`.
- **Gate: byte-identical `dictionary.jsonl`** (diff the files / compare a content hash). These runs never take
  the `energy_collapsed` branch, so the guard is always-false for them and output must be bit-for-bit
  unchanged. Any diff = the guard is mis-scoped → stop and re-examine.
- Cheap proxy already on hand: the 4 committed `runs/data/harvest_cal_*.csv` came from healthy runs; re-harvest
  after the fix and confirm identical.

### 5.3 Verification — the collapsed run is fixed end-to-end and still stops correctly

- Run `fail_repro` (heavy 5×10⁹) after the fix.
- Assert: still stops with `ENERGY_COLLAPSED` (code 51); its `dictionary.jsonl` now has **no negative Pb in any
  row** (the terminal garbage row is gone); the last row is the last *healthy* positive-Pb snapshot at the
  last valid `t`; row count drops by exactly one (the dropped reconciliation row) or the last row is replaced —
  document whichever the implementation yields.

### 5.4 Full suite

`pytest` green before and after; `pre-commit run --all-files` (ruff F-rules) clean. No new ruff scope.

## 6. Propagation checklist (the maintainer asked specifically that the result propagates)

Confirm each, post-fix:

1. `SimulationEndCode == 51` reaches the run metadata / terminal summary (it is set at line 1080, before the
   break, untouched by this fix). ✅ by construction.
2. `SimulationEndReason` string reaches the same. ✅ by construction.
3. The final `dictionary.jsonl` row carries a **finite, positive** Pb (last healthy snapshot), so any reader
   that takes `last_row.Pb` gets a physical value. ✅ via 5.3.
4. No downstream consumer relied on the *presence* of a reconciliation row for collapsed runs (grep
   `EndSimulationDirectly` / readers in `trinity/_output/`). **To verify during apply** — if a reader assumes a
   trailing reconciliation row exists, prefer the `Eb>0`-inside-block variant (§4) instead.

## 7. Apply order (when the maintainer green-lights production)

1. Write `test/test_energy_collapse_snapshot.py` (§5.1) → confirm it **fails** on current `main`.
2. Apply the one-line guard (§4).
3. Re-run §5.1 (now passes), §5.2 (byte-identical), §5.3 (fail_repro clean), §5.4 (full suite + ruff).
4. Commit dev branch `feature/PdV-trigger-term-pt2`; no production default changes; reconcile `INDEX.md`/`PLAN.md`
   and the `KMIX_PROTOTYPE.md` §2 diagnosis note (correct its line-ordering guess to point here).

*Written 2026-06-30 on `feature/PdV-trigger-term-pt2`. No production code touched.*
