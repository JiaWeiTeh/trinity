# R1 shadow-first — implementation plan (verified line-by-line, 2026-06-22)

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

**Goal:** build the R1 transition criterion (blowout `R2 > rCloud` + Eb-peak) into the production
1b transition site in **shadow mode** — it is *evaluated and logged every segment but never drives
the phase switch*. Production still terminates only on `cooling_balance`/`reached_tmax`/etc. The
**hard gate: the main `dictionary.jsonl` must be byte-identical** to pre-change (the shadow only
reads existing params and writes a separate sideline CSV).

## Verified facts (read against current source 2026-06-22 — re-verify per banner)

1. **The Eb-peak net energy is ALREADY computed in production.** `get_betadelta.py:426–434`:
   `L_gain = Lmech_total` (:426); `L_loss = bubble_props.bubble_LTotal` (:427) `+= bubble_Leak` (:432);
   `Edot_from_balance = L_gain − L_loss − 4·π·R2²·v2·Pb` (:434). This is exactly the PdV-inclusive
   net-energy / Eb-peak quantity. It is stored each segment as `residual_Edot2_guess`
   (`run_energy_implicit_phase.py:781–782`). ⇒ **Eb-peak criterion = `Edot_from_balance ≤ 0`** — no new
   physics, just a read.
2. `bubble_Lgain ≡ Lmech_total` and `bubble_Lloss ≡ bubble_LTotal + bubble_Leak` in 1b — so the live
   cooling ratio `(Lgain−Lloss)/Lgain` equals the offline `(bubble_Lgain−bubble_Lloss)/bubble_Lgain`
   used in H1/H2 (consistent).
3. **Blowout** = `R2 > rCloud`. `params['rCloud']` is `run_const` and always set in 1b (already read at
   `run_energy_implicit_phase.py:660`).
4. **Live transition site** = the terminator block, `run_energy_implicit_phase.py:1067–1132`; the
   `cooling_balance` decision is at **:1095** (`if Lgain>0 and (Lgain−Lloss)/Lgain < threshold: break`).
   This is exactly where a future *flip* would act, so the shadow check belongs here.
5. `betadelta_result` (and `params['residual_Edot2_guess']`) are in scope at the terminator. `R2`, `v2`
   are the terminator (post-ODE) loop vars; `Eb`, `Pb` via params. `path2output` is a resolved param
   (`registry.py:294`, always an mkdir'd dir) → safe sideline-file target.

## Design (minimal, inert, single edit to one file)

In `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py`:

1. **Tiny pure helper** (module-level, testable, reused by the future flip):
   ```python
   def evaluate_r1_shadow(R2, rCloud, edot_balance, k_blowout=1.0):
       """R1 transition criteria (SHADOW — does not drive). Returns (blowout, ebpeak)."""
       blowout = (rCloud is not None and rCloud > 0 and R2 > k_blowout * rCloud)
       ebpeak  = (edot_balance is not None and np.isfinite(edot_balance) and edot_balance <= 0.0)
       return bool(blowout), bool(ebpeak)
   ```
2. **Init before the loop** (near `termination_reason = None`, :622): `shadow_rows = []`,
   `shadow_blowout_t = None`, `shadow_ebpeak_t = None`.
3. **Shadow block at the terminator**, inserted immediately AFTER the `cooling_balance` check
   (after :1098, before the collapse-detection at :1101) — guarded so it can NEVER alter control flow:
   read `R2`, `params['rCloud'].value`, `params['residual_Edot2_guess'].value`; call the helper;
   append a row `{t_now, R2, rCloud, R2_over_rCloud, Eb, v2, Pb, bubble_Lgain, bubble_Lloss,
   edot_balance, blowout, ebpeak}`; on first True of each, set the `_t` tracker and `logger.info(
   "R1 shadow: <crit> would fire at t=… R2=… R2/rCloud=…")`. **No `break`, no `termination_reason`,
   no param writes.**
4. **Write the sideline CSV after the loop** (near the completion log ~:1200), wrapped in
   `try/except` (a sideline-IO failure must never affect the run):
   `<path2output>/shadow_R1_1b.csv`. Also log the two first-fire epochs in the completion summary.

**Why byte-identical:** the block only *reads* params already set this segment and appends to a local
list + writes a NEW file. It touches no physics param, no RNG, no control flow. The `cooling_balance`
decision at :1095 is unchanged.

## Gate & tests (rule-5 ladder)

- **G1 — byte-identical (hard):** run `simple_cluster` before vs after the edit; `dictionary.jsonl`
  must be byte-identical (`diff`/sha256). The sideline `shadow_R1_1b.csv` is the only new output.
- **G2 — unit test** `test/test_r1_shadow.py`: `evaluate_r1_shadow` truth table (blowout on/off,
  ebpeak on `edot<=0`, no-fire on `edot>0`/`nan`/`rCloud=None`).
- **G3 — regression:** `pytest -m "not stress"` green.

## Coverage of 1a + heavy clouds (offline supplement)

The in-code shadow lives in 1b (the transition phase / flip site). `fail_helix` collapses in **1a**, so
its Eb-peak is in 1a. To cover all phases for all configs, the analysis ALSO computes R1 firing
**offline from the full snapshot output** (Eb-peak = first `Eb` turnover; blowout = first `R2 > rCloud`),
and cross-validates the offline 1b segment against the in-code `shadow_R1_1b.csv` (must match). 1a
in-code instrumentation is deferred to the *flip* (it's not needed for shadow DATA).

## Data collection — all 8 configs (subagents, parallel)

Configs: 6 cleanroom (`docs/dev/transition/cleanroom/configs/*.param`) + `fail_repro`, `fail_helix`
(`docs/dev/failed-large-clouds/harness/params/`). One sim/process, `OMP_NUM_THREADS=1`, `timeout`-bounded,
parallel across the 4 cores (subagents in worktrees off the committed shadow branch). Each run emits its
`dictionary.jsonl` (for the byte-identical gate + offline R1) and `shadow_R1_1b.csv`. Persist all under
`docs/dev/transition/pt4/r1shadow/`.

## Plots & analysis (every finding a plot)

- `r1_firing_epochs` — per config: blowout epoch, Eb-peak epoch, vs the current (never-firing) cooling
  trigger, on a time axis; mark which fires first.
- `r1_vs_stall` — the cooling ratio (never <0.05) overlaid with where R1 *would* hand off.
- Heavy clouds: Eb-peak fires at/near birth (cross-check vs H4).
- Table `r1_shadow_summary.csv`: config, blowout_t, ebpeak_t, which-first, R2@handoff, v2@handoff.

## Out of scope (this shadow step)

No flip (production still switches on cooling_balance); no `transition_trigger` param yet (added at the
flip); no 1a in-code instrumentation; no momentum-continuation. Continuity of the handoff state into
phase 1c is the make-or-break check for the *flip*, not for shadow logging.
