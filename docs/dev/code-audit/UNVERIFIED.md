# UNVERIFIED — candidates that did not survive, or were never tested

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
>
> 🔗 **Cross-check the sibling docs — keep the workstream self-consistent.** This file is one of
> several living docs for its workstream (its `PLAN.md`, `FINDINGS.md`, `runs/README.md`, `NOTE_PATCHES.md`,
> and any other notes in the same folder). They drift out of sync *with each other* as fast as they drift
> from the code. Any agent or person editing one MUST, as part of the visit, circle back through the
> siblings and reconcile: if a number, status, claim, or line reference here contradicts a sibling — or a
> sibling has gone stale — fix it (or flag it, dated) so no two docs in the workstream disagree. Never
> update one in isolation.

**Status (2026-07-30):** 🔵 ACTIVE — the register the audit's method requires: candidates
kept **separate** from `FINDINGS.md` so a demoted claim can never be mistaken for a defect.

## Why this file exists

A multi-agent audit's dominant failure is confident nonsense. This file is the other half
of `FINDINGS.md`: everything raised as a defect that then failed verification, plus
everything still untested. **Nothing here should be acted on.**

Two of these were rated **S1** by their finders and would have been "fixed" without the
gate. One of those fixes would have broken working code.

---

## A. Removed by verification — do NOT act on these

### `S6-R-02` — "`pdot` is missing a factor of 2" · born **S1** · **REFUTED 3/3**

The physics premise is right (`pdot = 2L/v`); the assertion about the code is wrong.
`get_bubbleParams.py:308` is `Lmech / (2*np.pi*r**2*v_mech)` — the 2 cancels one of the 4
in `4π`. And `v_mech_total` is *defined* as `2*Lmech/pdot` (`update_feedback.py:181`), so
`4πr²·pRam ≡ pdot_total` identically. Measured ratio **1.00000000** across 0.1–5.0 Myr.

> ⚠️ **Applying the proposed fix would double the ram pressure.** The claim's likely
> origin is `get_bubbleParams.py:409`, a literal `Lmech_total / v_mech_total` that *looks*
> like the defect but is `ṗ/2`, with the 2 cancelled analytically in the inner-shock balance.

### `S8-R-01` / `SF-001` — "`odeint` reads uninitialised memory as physics" · born **S1** · **REFUTED 3/3**

The **mechanism is real** and was demonstrated more strongly than the finder did: a
heap-poison test returned 2697/3000 sentinel values. But `shell_structure.py:181-188`
truncates at the first `phi <= 1e-9` or mass-swept row, which across **416 induced
failures was always strictly before the failure row** — 0/416 garbage reads, outputs
bit-identical. And 0 failures in ~1000 realistic solves at the shipped `mxstep=50000`
(vs 416 at the pre-fix `mxstep=500`).

**Residual, kept in `FINDINGS.md` as S4:** `full_output=1` + an istate check is cheap
defence in depth.

### `S9-R-01` — "CIE cooling density factor wrong by up to 5.29×" · born **S1** · **CLEARED**

`ndens` is n_H and `chi_e` is n_e/n_H, so the branch correctly computes `n_e n_H Λ` —
the error-factor-1.00 case. A lens asserted `ndens` was total density; it was an
inference, and it was wrong.

### `S9-R-02` — "non-CIE branch short by n²" · born **S1** · **CLEARED (code)**

The cube is volumetric — measured `d(log cool)/d(log n) = 2.014` over 14 decades. The
docstring calling it `[erg cm3/s]` is wrong (**S3**, in `FINDINGS.md`) and is exactly the
kind that would license a future "fix" multiplying by n².

### `S13b-R-01` — "CLOUDY dlaw density missing a composition factor" · **CLEARED**

`shell_n_arr` is already n_H. **No factor should be added.**

### `S13b-R-02` — "`ZREL` wrong by ~1.85 dex" · **CLEARED**

`ZCloud` is declared `unit='Zsun'`, consumed as CLOUDY's linear solar-relative scale, and
`_validate_ZCloud` pins `Z == 1` anyway. (The lookup did surface a real, separate S2 —
the `--z-override` validator bypass — which is in `FINDINGS.md`.)

### `S13a-B-05` — "`is_successful_run` uses list membership" · **WITHDRAWN**

`trinity_reader.py:543` is `0 <= int(ec) <= 9`, a range test identical to `is_clean()`.
The docstring's `[0, 9]` is interval notation, which a prose-only lens read as a Python
list literal. Residual **S4** on the ambiguous notation.

### `S11-R-21` — "`__format__`/`__truediv__` missing" · **WITHDRAWN**

Both present in `dictionary.py`.

---

## B. Demoted — real, but smaller than claimed

| id | born | now | why |
|---|---|---|---|
| `S4-R-01` | S1 | **S2** | Mechanism confirmed and reachable, but the diagnosis was wrong: the energy equation is the canonical form; the defect is that the ODE and the `n_IF_Str` cap use two different values of `Pb`. Measured cost ~1 % (`ΔR2` 0.25 %, `Δv2` 0.81 %). |
| `SF-002` | S1 | **S2** | 600 production `fsolve` calls, 0 non-converged, ~230× basin margin. The claim's "returns the seed" is wrong for smooth non-convergence (returns the last iterate). The constant-residual and false-`ier=1` modes stand but neither lens could reach them. |
| `ST-001` | S1 | **S2** | Trajectory unaffected — phase 1b recomputes everything from `params` before integrating, and the one stale-`Pb` consumer returns exactly 0.0 at the default `coverFraction`. One wrong **output row** survives. Unreachable on the baseline (a 48-point grid gave zero cloud-boundary crossings). |
| `S11-R-06` | S1 | S3 | `cooling_balance` is an inline ratio test, not an ODE event; its factory is unpacked and never used. |
| `S11-R-09` | S1 | S4 | All four `apply_event_result` sites pass the event root time, not the last step. |
| `S11-R-10` | S1 | S4 | `EndSimulationDirectly` defaults to `False`, not `None`; sweeps are process-isolated. |
| `S11-R-07` | S2 | S4 | `Lgain` is guarded on both paths. |
| `S11-R-11` | S2 | S3 | `stop_at_rCloud_nSnap`'s `>= 1` semantics have no consumer. |
| `S12b-B-01` | S2 | S3 | The `~9.42e-58` `mu_convert` factor is comment-only; the code calls `convert2au('m_H')` = 8.4166e-58, correct to 1.00002. |

---

## C. Never tested — 8 S1-class candidates

Listed in `FINDINGS.md` §"raised but never gate-tested". They carry an S1 rating from a
reconciler or sweep and **nothing has tried to kill them**.

The prior from section A is the reason to hold them loosely: **of the 7 defects that were
panelled, 2 were removed outright and 3 were demoted.** Expect a similar fraction here.

**Two have since been tested and both were promoted out**, so the prior is not
one-directional:

- `S8-R-02` — dynamically confirmed by Phase 6 (`n_IF_Str == shell_n0` bit-identical).
- `S11-R-02` — dynamically confirmed **and widened**: the claimed false negative is real
  but largely masked in phases 1b/1c/2 by a redundant detector, while a *second*
  misclassification the claim never mentioned — `large_radius_event` latching
  `isCollapse=True` on an **expanding** shell — is unmasked. See `FINDINGS.md` §5.
  It is the first untested candidate to **grow** under verification rather than shrink.

---

## D. All 196 S2s are untested

The method calls for a skeptic panel on every S1 **and** S2. At the measured ~150k tokens
per skeptic that is ~88M tokens for the S2 tier alone, which is not proportionate. They
are carried in `data/findings_inventory.csv` at their reconciler-assigned severity and
should be read as **candidates**.
