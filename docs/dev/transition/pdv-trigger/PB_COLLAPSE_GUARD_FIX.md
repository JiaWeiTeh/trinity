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

**STATUS: APPLIED + tested (2026-06-30), maintainer-authorized.** The fix landed on
`feature/PdV-trigger-term-pt2` with its failing-first test, regression, and the full suite green (§5).
**The applied fix is two-part, not one-line** — skipping the reconciliation alone dropped the
`ENERGY_COLLAPSED` code from the output (it was the only snapshot that persisted the end code), so an
`else: params.save_snapshot()` was added to keep the stop fate propagating (see §4, the propagation correction
the test caught). It is a **correctness-preserving robustness change** to the energy-implicit phase,
**independent** of the κ_mix effort — a pre-existing output-hygiene bug surfaced while harvesting `Pb(t)` for
the κ_mix prototype.

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

**Skip the reconciliation *recompute* on the energy-collapse exit, but still persist a final snapshot so the
stop fate reaches the output.** Recomputing `Pb = (γ−1)·Eb/V` from the invalid post-collapse `Eb` produces
garbage; but the reconciliation `save_snapshot()` was *also* the only call that wrote `ENERGY_COLLAPSED` (code
51) into the dictionary (the in-loop snapshot at line 946 runs *before* collapse detection). So skipping the
whole block silently dropped the end code — **the failing-first test (§5.1) caught exactly this.** The applied
fix guards the recompute *and* adds an `else` that saves the last-healthy state (which still carries `Pb>0`
from line 868 and the end code set at line 1080):

```python
# phase1b_energy_implicit/run_energy_implicit_phase.py
if termination_reason != "energy_collapsed":
    try:
        feedback_final = get_current_sps_feedback(t_now, params)
        ...                                   # full reconciliation (recompute Pb/shell/forces, save)
        params.save_snapshot()
    except Exception as e:
        logger.warning(f"Phase-boundary reconciliation failed: {e}")
else:
    # Eb<0 -> compute_R1_Pb gives garbage; params still holds the last healthy Pb + the
    # ENERGY_COLLAPSED code/reason -> save them so the stop fate reaches the output.
    params.save_snapshot()
```

**Why this option over the alternatives:**

- ✅ **Chosen — skip the recompute on collapse, but still `save_snapshot()`.** Minimal, intention-revealing,
  and unchanged for every non-collapsing run (the guard is false only on the collapse path). The collapsed run
  keeps its last-valid Pb + the ENERGY_COLLAPSED code/reason. *(The `else: save_snapshot()` is essential — see
  §4-block above; without it the end code never reaches the output, which the failing test caught.)*
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

### 5.1 Unit — the collapse path short-circuits cleanly (the failing test) ✅ DONE

`test/test_energy_collapse_snapshot.py` runs a heavy collapsing cloud (`mCloud=5e9, sfe=0.1, nCore=1e2`,
collapses at t~3e-3 Myr, ~80 s) end-to-end via `run.py` (mirrors `test_run_smoke`) and asserts on the output
`dictionary.jsonl`:

- last row `SimulationEndCode == 51` — **collapse propagates** (this is what the propagation correction in §4
  restored; *the first fix attempt without the `else` failed this assertion → code `None`*).
- **no row** carries `Pb < 0` — *fails on current `main`* (terminal `Pb ≈ −1.6×10¹⁸`, row 52), *passes after*.
  The bug-→-failing-test→-fix gate (rule 4); confirmed red (1 failed, 82 s) then green (1 passed, 80 s).
- the last row's `Pb` is finite and **positive**.

### 5.2 Regression — the healthy path is unchanged (the "don't break anything" gate) ✅ DONE

The fix cannot perturb a non-collapsing run **by construction**: the only change is the `if termination_reason
!= "energy_collapsed":` wrapper (always True for a healthy run → the original block runs identically) plus an
`else` (never taken). Verified empirically: ran `mCloud=1e5, sfe=0.3, stop_t=0.03` (reaches & exits the
implicit phase, so the changed code path *is* exercised) **before** (stashed) and **after** the fix, in
separate processes.

- **Finding — trinity is NOT bit-reproducible run-to-run.** Pre-fix vs post-fix `dictionary.jsonl` differ, but
  **so do two *same-code* runs**, in *exactly* the same three keys: `F_ram_SN`, `Lmech_SN`, `pdot_SN` — the SN
  feedback terms at **~1e-22** (physically zero; no SN at t~3e-7 Myr), i.e. BLAS-threading noise. **All physics
  fields** (`Eb`, `Pb`, `R2`, `v2`, `T0`, `bubble_LTotal`, `shell_mass`, …) are **bit-identical** across runs.
- **So the gate is: "differs only in the 3 nondeterministic ~1e-22 SN-noise terms, identical to a same-code run
  pair; all physics bit-identical."** A literal byte-identical `dictionary.jsonl` is **unachievable even for
  the same code** (this corrects the original gate wording, which assumed bit-reproducibility). The fix passes:
  its diff set == the same-code diff set.

### 5.3 Verification — the collapsed run is fixed end-to-end ✅ DONE

Covered by 5.1's end-to-end run (a heavy collapsing cloud, the `fail_repro` regime): still `ENERGY_COLLAPSED`
(51), zero negative-Pb rows, terminal row = last healthy positive-Pb snapshot. The reconciliation row is now
*replaced* by the bare-snapshot row (same `t`, healthy Pb) rather than dropped, so the end code still lands.

### 5.4 Full suite ✅ DONE

`pytest -q` (default, non-stress): **596 passed, 4 deselected, 0 failed** (121 s) with the fix. Ruff F-rules
on the edited file: clean **on the changed lines** (9 pre-existing F401/F841 at lines 61–665 are untouched
dead code — flagged, not deleted, per CLAUDE.md rule 3).

## 6. Propagation checklist (the maintainer asked specifically that the result propagates)

Confirm each, post-fix:

1. `SimulationEndCode == 51` reaches the run metadata / terminal summary (it is set at line 1080, before the
   break, untouched by this fix). ✅ by construction.
2. `SimulationEndReason` string reaches the same. ✅ by construction.
3. The final `dictionary.jsonl` row carries a **finite, positive** Pb (last healthy snapshot), so any reader
   that takes `last_row.Pb` gets a physical value. ✅ via 5.1/5.3 (last row `Pb>0`, asserted).
4. A trailing snapshot row **still exists** on collapse (the `else: save_snapshot()`), so any consumer that
   relied on the *presence* of a final row is unaffected — only its `Pb` changed from garbage to healthy. ✅
   (This is why the `else` was added rather than dropping the snapshot outright.)

## 7. Apply order — ✅ EXECUTED (2026-06-30)

1. ✅ Wrote `test/test_energy_collapse_snapshot.py` (§5.1) → confirmed **red** on current code (negative Pb).
2. ✅ Applied the guard (§4) — and, after the test caught the dropped end code, added the `else:
   save_snapshot()` propagation branch.
3. ✅ §5.1 green; §5.2 equivalent (only the 3 nondeterministic ~1e-22 SN terms differ, same as a same-code
   run pair); §5.3 collapsed run clean; §5.4 full suite **596 passed**.
4. ✅ Reconciled `INDEX.md`/`PLAN.md` and the `KMIX_PROTOTYPE.md` §2 note. No production *default* changes
   (behaviour identical for every non-collapsing run).

*Written + applied 2026-06-30 on `feature/PdV-trigger-term-pt2`.*
