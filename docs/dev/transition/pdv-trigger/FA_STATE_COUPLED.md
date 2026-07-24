# FA_STATE_COUPLED — the state-coupled f_A: derive the density dependence instead of fitting it (single source of truth for the successor workstream)

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
> a committed artifact under `docs/dev/` (a CSV/table in `docs/dev/transition/pdv-trigger/data/`, or a
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

**Status (2026-07-22, created):** THE single plan doc for the **state-coupled f_A** workstream —
the successor the scalar-f_A stream's own Phase-6 tree names (row 3, `FINDINGS.md §15j`). Per the
maintainer's one-stream directive there are **no parallel plans**: the scalar-f_A history, evidence
and Phase 0–6 record stay in `SOURCE_TERM_DESIGN.md` (its §4 sketch is hereby **promoted here** and
must not be extended there); THIS doc plans only the successor. **Phase SC-0 (offline screen) is
OPEN and needs no ruling — it is read-only.** SC-1 onward are gated on the parent's Phase-6
maintainer ruling (§3 below). Nothing here touches production code until SC-1. **Phase-6 ruling
STARTED 2026-07-22: clause 1 RULED — f_mix RETAINED as an opt-in fallback, retirement deferred +
staged (R0→R2 ladder, §3); default stays `none` (clause 3). Clauses 2/4 (adopt scalar f_A as the
diagnostic knob; greenlight successor) await an explicit nod — but SC-0 may run regardless.**

## 0. Why this workstream exists (one paragraph)

The scalar f_A was measured to work — every clean L21b bench reaches the Θ band — but at a steeply
density-dependent dose: band entry **13.9 / 53.5 / 74.8** for n̄ = 5520/690/43 (spread 5.39×, fit
f_A(n̄) ≈ 315·n̄^−0.335; `FINDINGS §15j`), while f_mix was eliminated outright. A fitted f(n̄) is
exactly the kind of un-derived magic function this workstream's history warns against
(`INDEX.md §1.5`). Hypothesis to test: **the density dependence is not a free function but El-Badry
mixing-layer physics** — replace the scalar with f_A evaluated from the live bubble state via the
L_int closed form, leaving **one physical constant (λδv)** to serve the whole suite.

## 1. The object — definition + the design decisions SC-0 must freeze

Candidate definition (the one-read swap at the two production edit sites,
`bubble_luminosity.py:435/845`):

```
f_A_state(t) = L_int^EB(R2, Pb; λδv) / (L2+L3)_resolved^(prev accepted step)
L_int^EB     = 4π·√(α·λδv) · R2² · Pb^(3/2) · √Λ(T_pk) / (k_B·T_pk),   T_pk ≈ 2×10⁴ K
```

(`ELBADRY_REFERENCE.md §7` option B — the direct form; §9 verified TRINITY's plumbing realizes the
El-Badry budget faithfully.) The knob's free physical constant is **λδv only** (literature anchor
λδv ≈ 3–3.5, `LANCASTER_REFERENCE.md §6`); the goal is a single λδv across the suite where the
scalar needed 14→75.

**Decision points SC-0 freezes (recommendation first, decide before SC-1):**

| # | decision | recommendation | why |
|---|---|---|---|
| D1 | closure lag | **previous accepted step** (lagged) | precedent: the §4-sketch q_w closure; avoids a new inner iteration in the hot loop |
| D2 | option A (n-mapped θ(λδv, n_amb)) vs **option B (direct L_int)** | **B** | no n-mapping; saturation emerges via Pb; robust at the early-core/late-blowout extremes where A diverges (`ELBADRY_REFERENCE §7`) |
| D3 | caps/floors | clamp f_A_state ∈ [1, f_cap≈256]; unchanged T<10^5.5 band gating; if (L2+L3)_prev → 0 (dense collapse) hold last finite value | keeps the dense-collapse stiffness (bench5_fa16_diag freeze, `§15h`) from amplifying; 256 = 2× the largest measured need |
| D4 | knob surface | sentinel string **`cooling_boost_fA='elbadry'`** resolved like `cooling_boost_kappa='auto'` | no new param; single-knob validator extends naturally; default '1.0' stays byte-identical |

## 2. Phase ladder (solver hot loop ⇒ full rule-5 ladder; gates BEFORE code)

**SC-0 — offline screen (read-only; MAY run pre-ruling). THE falsification gate.**
1. *Data prerequisite:* the committed traj CSVs lack Pb (`t_now,theta,Lcool,Lleak,Lmech,R2`), but
   every `dictionary.jsonl` logs `Pb` (reader vocabulary, `trinity_reader.py:165`). Either
   (a) maintainer re-harvests the bench1/2/3 fa1(+fa16) diag arms on Helix with a Pb+L2+L3 column
   set (extend `harvest_bench5.py` with `--extra-cols`), or (b) run the 3 diffuse fa1 diag arms
   locally (~20–45 min each, walltime evidence `data/bench5_durations.csv`) and harvest there.
   Commit as `runs/data/bench_state_traj/`.
2. *Offline calculator* `data/make_fa_state_screen.py`: evaluate f_A_state(t) along those
   trajectories for λδv ∈ {1, 2, 3, 3.5, 5}; compare its blowout-window average against the
   **measured** band-entry doses 13.9/53.5/74.8. **PASS:** one λδv reproduces all three within a
   factor ~2 (the same tolerance the p=3.33 law achieved). **FAIL:** no λδv does ⇒ the closed form
   does NOT derive the curve ⇒ stop, record, and the fitted f_A(n̄) remains the honest shipped
   result — no production code gets written.
3. Persist: `data/fa_state_screen.csv` + figure; register in REPRODUCE.

**SC-1 — wiring (gated on Phase-6 ruling + SC-0 PASS).** The one-read swap at the two edit sites +
registry sentinel resolver + validator extension + `test_fA_state_coupled.py`. Default `'1.0'`
stays the LITERAL float path (byte-identity preserved by construction, same guard style as today).

**SC-2 — gates (rerun the parent's Phase-3 pattern).** (i) default LITERAL byte-identity
(`dictionary.jsonl` sha256, pre==post); (ii) per-call equivalence: live f_A_state values vs the
SC-0 offline calculator on captured states; (iii) live sign checks (dMdt falls, θ rises, no
freeze/no-root regressions on the stiff edges `f1edge_*`).

**SC-3 — matrix (Helix).** The 9 theta5s configs × λδv {2, 3, 3.5, 5} = **36 arms** (no dose grid —
that is the point). Fire map + controls (`fail_repro`, `small_1e6` must stay cold).

**SC-4 — THE acceptance gate (Helix).** The 5-bench L21b suite × the same λδv grid, prod+diag =
**40 arms**, same blowout-window Θ_cum metric and harness as bench5/6. **PASS:** a single λδv
(target ≈3±1) lands Θ_cum ∈ [0.90, 0.99] on bench3/2/1 simultaneously, dense benches still fire,
controls cold, dex-vs-EB improves on the scalar's ≥0.85. **FAIL:** record how close (dex per
bench), keep the knob diagnostic-only.

**SC-5 — ship decision.** SC-4 PASS ⇒ the default-flip ruling package (one derived knob, one
physical constant, L21b-validated — the paper's f_A story completes). SC-4 FAIL ⇒ documented
negative + the scalar f_A(n̄) table stands as the calibration.

## 3. Phase 6 of the PARENT — the RULING (source of truth; SC-1+ waits on this)

This is THE Phase-6 ruling of record (this doc is the single place it lives; `SOURCE_TERM_DESIGN.md
§3 Phase 6` and any `FINDINGS.md §15k` point HERE, they do not restate it). Clauses tagged
**[RULED 2026-07-22]** are the maintainer's decision; **[pending]** clauses still want an explicit nod.

1. **f_mix — RETAINED as an opt-in fallback [RULED 2026-07-22, maintainer].** bench6 eliminated f_mix
   as a *calibration* knob (never reaches the L21b band ≤8, wrong-sign dose-response on the diffuse
   benches, fm8 false-fires — `FINDINGS §15j`) — but NOT as a *fallback*. It stays fully wired and
   supported for now: it is a valid opt-in mechanism AND the control arm the bench harness relies on.
   Nothing is removed while f_A is not yet the production path. **Retirement is deferred and STAGED
   (the "safely and slowly" ladder), each rung gated on the one before:**
   - **R0 (now):** `cooling_boost_mode='multiplier'` retained, opt-in, **inert by default** (default
     is `none`). Registry `info` for `cooling_boost_mode`/`cooling_boost_fmix` gains one line:
     "fallback — superseded for L21b *calibration* by f_A (`FA_STATE_COUPLED.md`); retained pending
     the state-coupled f_A shipping." No behavior change. (This is the only code touch clause 1
     authorizes now — a doc-string edit, byte-neutral.)
   - **R1:** only AFTER the state-coupled f_A ships as the production default (SC-5 PASS) AND ≥1
     release cycle of it running clean — mark `multiplier` **deprecated** in the registry (still
     works, emits a load-time deprecation warning). No removal yet.
   - **R2:** after ≥1 further cycle with nothing in-repo relying on it (grep the params/tests/docs),
     remove the `multiplier` branch — per the project rule, `git mv` the code + its arms into
     `docs/dev/to-be-removed/` for maintainer review, never a direct delete.
   - **Abort rule:** if the state-coupled f_A does NOT ship (SC-4/SC-5 FAIL), the ladder STOPS at R0
     — f_mix stays indefinitely and the retirement is void.
2. **Scalar f_A — calibrated *diagnostic* knob [pending explicit nod].** The measured f_A(n̄) table
   (13.9/53.5/74.8; ≈315·n̄^−0.335, HPC provenance `§15j`) is quotable; NOT proposed as a production
   default (the successor supersedes it). Recommended adopt.
3. **Production default — UNCHANGED [RULED 2026-07-22, implied by clause 1].** `cooling_boost_mode=none`,
   `cooling_boost_fA=1.0`, byte-identical. Keeping f_mix as an opt-in fallback presupposes no default
   flip now — so this is settled by clause 1.
4. **Successor — greenlit [pending explicit nod].** This workstream proceeds; **SC-4 is the pre-agreed
   bar** for any future default flip (and, per clause 1, the trigger that starts the f_mix R1 rung).
   SC-0 (offline falsification screen) is read-only and may start immediately regardless.

Parent loose ends that stay in the PARENT's ledger (not this doc): the dMdt reducer re-run on the
Helix theta5s raw arms (`§15e` residue); Fig-17 re-digitization before quantitative fits; V_w
[I]-grade; `rosette-cf/figs/README.md` banner (other workstream).

## 4. Artifacts & reconciliation

Artifacts this plan will create: `runs/data/bench_state_traj/` (SC-0 data),
`data/{make_fa_state_screen.py, fa_state_screen.csv}` (SC-0), the SC-1 diff + tests, SC-3/SC-4
params + summaries under `runs/params/{sc_matrix,sc_bench}/` + `runs/data/`, REPRODUCE rows on
landing. Siblings to keep reconciled on every edit: `SOURCE_TERM_DESIGN.md` (§4 pointer + Phase-6
ruling), `FINDINGS.md` (new §15k+ entries), `INDEX.md` (this workstream's row), `PLAN.md` ledger,
`ELBADRY_REFERENCE.md`/`LANCASTER_REFERENCE.md` (imprints — read-only anchors here).
