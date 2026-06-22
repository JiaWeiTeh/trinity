# html-insights — storyline merge: verification summary & fix-ledger

> ⚠️ **This document may be out of date — verify before trusting it.** Point-in-time
> audit, not a maintained spec; re-check each claim against current source.
>
> 🔄 **Living ledger — recheck and refine on every visit.** Re-run the verdicts when you
> touch the relevant code; tick the fix boxes as they land.
>
> 💾 **Persist diagnostics — commit, don't re-run.** The per-report ledgers (01–05) carry the
> `file:line` evidence so a future visit need not re-derive it.

**Verified:** 2026-06-22 · **Method:** five parallel read-only agents, each checking one or two
rendered HTML reports **line-by-line against current `trinity/` source** (HEAD of
`claude/exciting-gates-mkxqn6`), cross-checked with the committed CSVs/harnesses and the
per-workstream `.md` docs. Per-report evidence: `01_betadelta.md` … `05_tables.md`.

This pass supports merging the per-workstream reports into four storyline "books"
(`build_storylines.py`). Sources are the source of truth — fixes land in each report's
generator (or the saved HTML for betadelta) and the book is recomposed.

## Scorecard

| Report (chapter) | Storyline | ✅ | ⚠️ | ❌ | ❓ | Verdict |
|---|---|--:|--:|--:|--:|---|
| `betadelta` (new upload) | S1 ch1 | 32 | 6 | 5 | 3 | richest + most stale; fix before publish |
| `F1_REPORT` | S1 ch2 | 14 | 2 | 0 | 2 | accurate; 2 numbers unbacked → caveat |
| `transition_report` | S1 ch3 | 32 | 3 | 0 | 1 | clean |
| `shell-solver/insights` | S2 ch1 | 11 | 1 | 0 | 0 | §6 "what shipped" stale/incomplete |
| `failed-large-clouds/insights` | S3 ch1 | 32 | 6 | 0 | 1 | fix shipped exactly; line drifts |
| `magic-numbers/tclamp_report` | S4 ch1 | ✓ | 2 | 0 | 0 | clean |
| `cooling/refactor-audit.md` | S4 ch2 | — | — | — | — | "nothing shipped" stale (2 shipped) |

## Cross-cutting finding (affects multiple docs)

**The β–δ Phase-4 default flip SHIPPED.** `betadelta_solver` default is now **`hybr`**
(`trinity/_input/registry.py:307`, `trinity/_input/default.param:49`) — not `legacy`. This:
- falsifies the betadelta report's "default still legacy" (❌, §4/§5/§13/timeline);
- makes `DOC_STATUS.md:34,65–67,78` and `README.md:86` stale (they call the flip a "deferred
  open tail");
- strengthens the S1 arc: the transition stall (FINDINGS: 0/6 reach momentum under hybr) is the
  behavior of the **production default**.

**pshadow / P0 are SUPERSEDED, not vindicated** (the question you asked). pshadow §1 assumed
"flat configs transition by cooling (F0 fires @0.197)"; the clean-room FINDINGS §3 found F0 never
fires in any of 6 regime-spanning configs (flat `simple_cluster` blows out geometrically at
t≈0.09 while F0 floors at 0.40). The narrower P0 harvest (`transition_dense_flat.csv`) did fire
F0, but the generalisation was premature. Nothing shipped (`git grep transition_trigger|blowout|shadow trinity/` → empty). → **one-line historical mention, no chapter.**

## Master fix-list (tick as applied)

### Critical (substance / "what shipped")
- [ ] **betadelta**: "default still legacy" → **`hybr` (Phase 4 shipped)** — every occurrence (§4/§5/§13/timeline). Evidence: `registry.py:307`, `default.param:49`.
- [ ] **shell §6**: add the `_NSHELL_MAX=1e120` clip guard (`get_shellODE.py:32,100`, commit `b27cede`) — the fix that actually stops the LSODA `t+h=t` flood; `mxstep=50000` (`shell_structure.py:35`, `00e9f54`) only silences a separate Python `ODEintWarning`. `MIGRATION_PLAN.md` retracts the mxstep-fixes-the-flood claim. Regenerate `insights.html`.
- [ ] **DOC_STATUS.md / README.md**: mark the Phase-4 hybr default flip **shipped** (drop "deferred open tail").

### Stale line/path references (betadelta, from `01_betadelta.md`)
- [ ] velocity ODE cite `bubble_luminosity.py:1150` → **`:411`** (file is 1083 lines).
- [ ] cooling-integral cites `:612/659/677` → integrand/L defs **`:696` (L_bubble), `:745` (L_conduction), `:785` (L_intermediate)**.
- [ ] §5 Phase-3 table: `steep·hybr` β-range `−2.44→2.82` → **`0.59→2.82`** (β_min −2.44 is `sweep_steep` 4-Myr, `PHASE2_ARMS.md:223`, not the 3-Myr `steep·hybr` `:198`).
- [ ] §3 footnote: predictor/consistency test `HYBR_PLAN.md:52,49–58` → **`:74–83`**.
- [ ] §14 paths: `analysis/PHASE0_BETADELTA_BASELINES.md`→`docs/dev/archive/betadelta/PHASE0_BASELINES.md`; `analysis/BETADELTA_PHASE2_ARMS.md`→`…/PHASE2_ARMS.md`; `analysis/stalling-energy-phase.md`→`…/stalling-energy-phase.md`; `docs/dev/BETADELTA_HYBR_PLAN.md`→`…/HYBR_PLAN.md`; `scratch/phase2/`→`…/diagnostics/`; `scratch/phase6/`→`…/velstruct/`.

### Caveats / lower priority
- [ ] **F1**: caveat the Era-A wall-time `222.7→199.6 s` (`:71`) and `~2.3× full-run` (`:162,253–254`) as indicative, not CSV-backed (the only full-run CSV `ab_fullrun.csv` is BUGGED — in-process global-state leak; the report already flags this in its "false alarm" box). The per-call ~1.5× IS backed (`master_p0_table.csv`). Regenerate.
- [ ] **F1** ⚠️: `edge_hidens` worst rel-diff is `6.1e-5` (v2), report cites `~6e-6` (Eb). Minor.
- [ ] **largeclouds** (code/doc hygiene, out of the HTML): `get_bubbleParams.py:230` inline comment still "Catastrophic-cooling degeneracy" (relabel `f63c0e9` missed it); `misc/TERMINATION_EVENTS.md` lacks `ENERGY_COLLAPSED` (code 51, `simulation_end.py:90`).
- [ ] **cooling/refactor-audit.md**: "nothing shipped" → note the 1e4-floor fix (`cc8ae76`) and `NameError→ValueError` (`3deec3d`) shipped; core PR-1–4 still pending. Line refs drifted 5–45.

### Trivial (cosmetic line drifts, optional)
- tclamp: floor cite `:85`→`:130–131`; ZCloud `:88/148`→`:163/186`.
- largeclouds: G fix `:226-235`→`:228-235`; phase-1a F `~:313`→`:340`; phase-1b F `~:1006`→`:1007`.
