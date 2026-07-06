# R1 transition — shadow + opt-in keyword: findings

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

**Date:** 2026-06-22. **Plan:** `../R1_SHADOW_PLAN.md`. **Status:** shadow + opt-in keyword
**shipped to the branch** (default-off, byte-identical); driving validated on `simple_cluster`.

## What shipped

R1 = the energy→momentum transition built on events that *actually occur* under hybr (vs the
cooling-balance trigger, which never fires — pt4 README). Two criteria, in `run_energy_implicit_phase.py`:
- **blowout** `R2 > rCloud`; **Eb-peak** `Edot_from_balance ≤ 0` (the PdV-inclusive net energy,
  already computed in production at `get_betadelta.py:434` — the shadow only *reads* it).
- A pure, tested helper pair: `evaluate_r1_shadow()` (criteria) + `parse_transition_triggers()` /
  `r1_transition_decision()` (the keyword → which criterion drives).

**Two modes:**
1. **Shadow (always on, inert):** every 1b segment the criteria are evaluated, the first firing logged,
   and a sideline `shadow_R1_1b.csv` written. It never sets `termination_reason` / breaks / writes a
   physics param ⇒ the main `dictionary.jsonl` is **byte-identical**.
2. **Drive (opt-in):** the `transition_trigger` param (comma-separated set; default `cooling_balance`).
   Non-default sets (`blowout`, `ebpeak`, `r1`, or e.g. `cooling_balance,blowout`) make R1 *end the
   energy phase*, and `main.py` proceeds to 1c→momentum (same path as cooling_balance).

## Gates (all passed)

| gate | result |
|---|---|
| **G1 byte-identical** (default) | `dictionary.jsonl` sha256 `830b691a…` identical with vs without the shadow, and with the keyword added (`GATE_RESULT.txt`, two independent re-gates) |
| **G2 unit** | `test/test_r1_shadow.py` **14/14** (criteria + parse/alias/validation + decision) |
| **G3 regression** | `pytest -m "not stress"` **588 passed** |
| **Drive end-to-end** | `transition_trigger=blowout` on `simple_cluster`: R1 fired → **phase 1c (2.1 s) → momentum** (clean hand-off) |

## Live shadow data — where R1 would hand off (all 8 configs)

From `r1_shadow_summary.csv` (in-code shadow) and `shadow_<config>.csv`:

| config | blowout fires @ (Myr) | R2/rCloud | Eb-peak in-cloud? | first |
|---|---|---|---|---|
| small_dense_highsfe | 0.0117 | 1.02 | no | blowout |
| simple_cluster | 0.0902 | 1.02 | no | blowout |
| midrange_pl0 | 0.392 | 1.01 | no | blowout |
| pl2_steep | 0.840 | 1.05 | no | blowout |
| be_sphere | 0.856 | 1.01 | no | blowout |
| large_diffuse_lowsfe | 3.66 | 1.00 | no | blowout |
| fail_repro / fail_helix (5e9) | — (empty 1b) | — | (Eb-peak is a **1a** event) | — |

- **Blowout fires for every in-cloud config**; **Eb-peak never fires in-cloud** for normal clouds
  (`Edot_from_balance` stays positive — monotonic Eb, consistent with the pt4 result). So for normal
  clouds **blowout is R1's operative criterion**; the Eb-peak covers the heavy-cloud (5e9) end, which
  is a **phase-1a** event (the 1b shadow is empty because production's `Eb≤0` stop precedes the 1b
  shadow site — the heavy-cloud Eb-peak/collapse happens in 1a).
- **cooling_ratio min across all rows = 0.283** (never near 0.05) — the current trigger never fires.
- **Cross-validation:** the offline blowout epoch (first snapshot `R2>rCloud`) matches the in-code
  `blowout_t` to **|Δt| = 0** for the configs checked. So the figure preview = the live result.

## Figure
`../figures/r1_firing_preview.{png,pdf}` — per config, where R1 hands off (★ blowout / ◆ Eb-peak) vs the
grey bar = how long the current trigger keeps it energy-driven (never fires). Live-confirmed.

## Conclusion & caveats
- R1 gives a **defensible, finite energy→momentum transition for every in-cloud config**, where the
  cooling-balance trigger gives none — and it's **opt-in** (`transition_trigger`, default unchanged,
  byte-identical), so it commits nothing until selected.
- **Validated so far:** the *drive* hands off cleanly on `simple_cluster` (1b→1c→momentum). **Not yet
  validated:** the drive on the other configs and especially the **heavy-cloud Eb-peak** hand-off into
  1c (the Path-2 continuity question) — that is the remaining make-or-break for *using* R1 as a default,
  and is future work. The 1b shadow does **not** cover the 1a heavy-cloud Eb-peak (a 1a hook would be
  needed for the flip there).

## Artifacts (this folder)
`r1_shadow_summary.csv`, `shadow_<config>.csv` (8), `DATA_NOTE.md` (exact commands/stop_t), `GATE_RESULT.txt`
(byte-identical gate), `gate_simple_cluster_shadow_R1_1b.csv`. Production: `run_energy_implicit_phase.py`
(helpers + shadow + opt-in drive), `registry.py`/`default.param` (`transition_trigger`),
`test/test_r1_shadow.py`.
