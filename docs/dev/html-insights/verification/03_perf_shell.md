# Verification ledger 03 — F1_REPORT.html (A) + shell-solver/insights.html (B)

Verified 2026-06-22 on branch `claude/exciting-gates-mkxqn6`.

---

## SUMMARY

### (A) F1_REPORT.html — bubble_luminosity performance evolution (6-claim scorecard)

| Count | Meaning |
|-------|---------|
| ✅ 14 | Claims verified against current source or committed CSVs |
| ⚠️  2 | Correct in substance but with minor numeric discrepancy or missing context |
| ❌  0 | Flat-out wrong |
| ❓  2 | Untraceable — no committed artifact; number appears only in hardcoded HTML generator |

**Most important "did it actually ship as described?" finding:** All four eras (A–D / F1–F2)
have shipped exactly as described. The bubble-structure solve is `solve_ivp(LSODA, dense_output=True)`
via `_solve_bubble_structure` (no `odeint` call exists); `_CONDUCTION_NPTS = 2000` samples the
dense output in the conduction zone; `_RESIDUAL_NPTS = 500` replaces the 60k resample in the
hot residual. Commits `a245c29`, `1eb7f4d`, `5f4f229`, `76921f7`, `4a13075`, `24c6914` all
exist and touch the described files. The two ❓ items are the Era A wall-time numbers
(222.7 → 199.6 s) and the ~2.3× full-run speedup for `simple_cluster` — both appear only in
`make_f1_report.py` (hardcoded) with no committed CSV to reproduce them without re-running.

---

### (B) insights.html — shell-solver ODE investigation (6-claim scorecard)

| Count | Meaning |
|-------|---------|
| ✅ 11 | Claims verified against current source or committed CSVs |
| ⚠️  1 | Correct but incomplete — the mxstep fix description misrepresents the full shipped state |
| ❌  0 | Flat-out wrong |
| ❓  0 | Untraceable |

**Most important "did it actually ship as described?" finding:** The report's §6 "Solution &
what shipped" is **incomplete and partially stale**. It describes only the `mxstep=50000` fix
(`shell_structure.py`, commit `00e9f54`) but omits the second, later fix: the `_NSHELL_MAX=1e120`
clip guard in `get_shellODE.py` (commit `b27cede`, same day 2026-06-18). The `MIGRATION_PLAN.md`
(updated 2026-06-18) explicitly retracts the "mxstep fixes the user-visible warning" claim,
noting that `mxstep` silences the Python `ODEintWarning("Excess work")` but does NOT silence
the actual LSODA Fortran `t+h=t` flood, which is fixed by the clip guard in `get_shellODE.py`.
Both fixes are present in the current source. The report's diagnostic conclusions (equivalence,
no-speedup verdict, energy-phase-only event win) are all correct and CSV-backed.

---

## (A) F1_REPORT.html — claims table

| Claim | Report § | Evidence (file:line / CSV / commit) | Verdict | Fix needed |
|-------|----------|-------------------------------------|---------|------------|
| Bubble structure ODE now uses `solve_ivp(LSODA, dense_output=True)` via `_solve_bubble_structure`; `odeint`/`_odeint_checked` gone | §1.3, §5 | `trinity/bubble_structure/bubble_luminosity.py:417–478`; zero `odeint(` calls in file | ✅ | — |
| `_CONDUCTION_NPTS = 2000` — conduction zone sampled from dense output at K=2000 | §2, Era B | `bubble_luminosity.py:103`; used at line 717 | ✅ | — |
| `_RESIDUAL_NPTS = 500` — residual integrates on 500-point coarse `t_eval`; no dense resample | §4.6 (F1) | `bubble_luminosity.py:95`; `_get_velocity_residuals:329` (`t_eval=np.linspace(r2Prime_val,R1,_RESIDUAL_NPTS)`) | ✅ | — |
| Commits `a245c29`, `1eb7f4d`, `5f4f229`, `76921f7` (Era A) exist and touch the described files | §1.4 | `git log --all`: all four hashes present; `76921f7` removes dead odeint helper | ✅ | — |
| Commit `4a13075` (F2 free wins) exists | §3 | `git log --all`: `4a13075 perf(hotpath): F2 free wins` | ✅ | — |
| Commit `24c6914` (F1) exists | §4.6 | `git log --all`: `24c6914 perf(F1): drop the 60k dense-output resample` | ✅ | — |
| `7f08e58` drops misleading `_legacy` suffix | §3 | `git log --all`: `7f08e58 refactor(bubble): drop misleading '_legacy'` | ✅ | — |
| F2.3+F2.4: `get_dudt` cooling-cutoff cache yields +23.1%/call, bit-identical | §3 | `docs/dev/performance/HOTPATH_PLAN.md` Results ledger; `harness/verify_getdudt_equiv.py` | ✅ | — |
| F2.5 (`pdotdot_total` removal) DROPPED — not bit-identical | §3 | `HOTPATH_PLAN.md` F2.5 row: "⛔ dropped … feeds phase-1b RHS" | ✅ | — |
| Per-call speedup ~1.5× for M500 across 6 configs; accuracy ≤3.1×10⁻⁶ | §4.3, §4.4 | `docs/dev/performance/data/master_p0_table.csv`: M500 speedup 1.50–1.68× across configs; worst rel_dMdt 1.817e-06 | ✅ | — |
| Full-run equivalence: simple_cluster worst rel-diff 5.7×10⁻⁸ (R2/Eb/rShell); 251 pts over [0, 4.54] Myr | §4.5, §4.6 | `data/f1edge_matched_comparison.csv`: `f1cmp_simple,R2,4.5395,251,5.655e-08` | ✅ | — |
| edge_lowdens worst rel-diff 6.5×10⁻⁹; common range [0, 3.734] | §4.5 | `data/f1edge_matched_comparison.csv`: `f1edge_lowdens,R2,3.7340,238,6.477e-09` | ✅ | — |
| edge_hidens worst rel-diff ~6×10⁻⁶ (R2/Eb/rShell); common range [0, 0.052] | §4.5 | CSV shows `f1edge_hidens,Eb=6.016e-06`, `R2=2.086e-07`, `v2=6.130e-05`. Report states 6.0×10⁻⁶ citing "R2/Eb/rShell" — correct for Eb, omits v2 which is 6.1×10⁻⁵. Common range: CSV 0.0518 vs report 0.052 (rounding). | ⚠️ | Report should cite v2=6.1×10⁻⁵ as the true worst; and 0.052→0.0518. Not wrong; just incomplete. |
| Era A wall-time: residual-solve 222.7 s → 199.6 s (−10.4%) | §1.4 | `harness/make_f1_report.py:71` (hardcoded literal); no committed CSV with this measurement. `HOTPATH_PLAN.md:57` notes "the F1 '~21 ms vs 0.8 ms' timings remain subagent-microbenchmark-sourced". Era A timing is from the same era. | ❓ | Persist a CSV with this run timing so future sessions can verify without re-running. |
| ~21 ms resample vs ~0.8 ms integration (microbenchmark) | §4.1 | `HOTPATH_PLAN.md:170`: "⚠️ Caveat: the 21 ms is a *trivial-RHS* microbenchmark" — noted as subagent-sourced, not from a real-RHS committed CSV | ❓ | Persist a real-RHS committed microbenchmark CSV. |
| ~2.3× full-run speedup on simple_cluster | §4.6, §5 | `ab_fullrun.csv` is labeled BUGGED (in-process global state leak); no committed separate-process simple_cluster full-run wall-time CSV. `F1_SUMMARY.md:97` claims ~2.3× but cites the A/B which is the bugged run. The ~1.5× per-call speedup on simple_cluster is CSV-backed (`master_p0_table.csv`). | ⚠️ | Persist a committed separate-process wall-time CSV for simple_cluster to back the 2.3× claim. |

---

## (B) insights.html — shell-solver claims table

| Claim | Report § | Evidence (file:line / CSV / commit) | Verdict | Fix needed |
|-------|----------|-------------------------------------|---------|------------|
| Shell ODE solved with `scipy.integrate.odeint` (LSODA) over ~1000-pt grid (not migrated to `solve_ivp`) | §1 Background, §6 | `trinity/shell_structure/shell_structure.py:165,324`: two `odeint(...)` calls both use `_SHELL_ODE_MXSTEP` | ✅ | — |
| `_SHELL_ODE_MXSTEP = 50000` module constant, both `odeint` calls pass it | §6 shipped | `shell_structure.py:35` (`_SHELL_ODE_MXSTEP = 50000`); used at lines 167 and 326 | ✅ | — |
| H1 confirmed: all LSODA variants match `odeint` to ~10⁻⁸; Radau/BDF drift to ~10⁻⁷ | §4 Step 1 | `data/master_table.csv`: V_lsoda_teval worst_rel_n=2.6e-9…1.0e-8; V_radau_teval 3.9e-8…1.7e-7 | ✅ | — |
| H2 & H3 falsified: no variant faster than `odeint` overall; drop-in ~0.15–0.21×; Radau/BDF ~0.05× | §4 Step 2 | `data/master_table.csv`: V_lsoda_teval speedup_med 0.16–0.25×; V_radau_teval 0.02–0.07× | ✅ | — |
| LSODA+event wins 4.2–4.4× in degenerate energy phase; collapses to ~0.5× in implicit | §4 Step 3 | `data/master_table.csv`: sfe0.3 energy V_lsoda_event 4.18×; sfe0.3 implicit 0.53× | ✅ | — |
| 15-sample pass mis-read implicit as fully ionised (5.65×); 100-sample reveals mixed (0.53×) | §4 Step 3 | `MIGRATION_PLAN.md` §P0-matrix ("At 15 implicit samples … the event scored 5.65×. At 100 samples … collapses to 0.53×") | ✅ | — |
| sfe0.3 default energy: odeint_ms=9.54 ms, 100% excess-work, 0% mass-limited | §5 master results | `data/master_table.csv`: `sfe0.3,energy,20,20,0,9.541,1.0,0.0` | ✅ | — |
| sfe0.3 default implicit: odeint_ms=0.92 ms, 58/42 ion/neu, 42% mass-limited | §5 master results | `data/master_table.csv`: `sfe0.3,implicit,100,58,42,0.922,0.15,0.42` | ✅ | — |
| `odeint(mxstep=50k)` variant: bit-identical (rel_n=0), speed ~1.0× in realistic configs | §4 Step 5, §6 | `data/master_table.csv`: V_odeint_hi: worst_rel_n=0.00e+00, speedup_med=0.98–1.01× | ✅ | — |
| Radau/BDF fail (ok < total) in degenerate configs | §5 master results | `data/master_table.csv`: sfe0.3 energy V_radau_teval `ok=0,n=20`; V_bdf_teval `ok=0,n=20` | ✅ | — |
| Harness artifacts committed under `docs/dev/shell-solver/` on branch `bugfix/LSODA-shellODE` | §7 Artifacts | `git log --all` shows commits `00e9f54`, `b27cede` etc. all from PRs merging `bugfix/LSODA-shellODE` | ✅ | — |
| §6 "what shipped": only `mxstep=50000` in `shell_structure.py` described; `_NSHELL_MAX` clip guard in `get_shellODE.py` omitted | §6 | Current source has TWO shipped fixes: (1) `shell_structure.py:35` `_SHELL_ODE_MXSTEP=50000` (commit `00e9f54`, 2026-06-18 08:42) — fixes Python `ODEintWarning("Excess work")`; (2) `get_shellODE.py:32` `_NSHELL_MAX=1e120` + `min()` at line 100 (commit `b27cede`, 2026-06-18 18:03) — fixes the actual LSODA Fortran `t+h=t` flood. `MIGRATION_PLAN.md` (updated 2026-06-18) explicitly retracts the "mxstep fixes the user-visible warning" conclusion and redirects to `OVERFLOW_FIX_PLAN.md`. The HTML was generated (commit `5f4f229` era) before the clip guard shipped. | ⚠️ | Regenerate `insights.html` (run `python docs/dev/shell-solver/make_insights_html.py`) after updating §6 to include the `_NSHELL_MAX` clip guard as the primary shipped fix and noting `mxstep` silences the separate Python `ODEintWarning`. |
