# Where `gamma_adia` has been absorbed into a numeric literal

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

**Status (2026-08-08):** 🔵 ACTIVE — the systematic sweep for γ that has been
cancelled into a numeric constant at γ=5/3. Supersedes sweep ②'s unverified
"Rahner A12 pair" phrasing: it is a **triple**, and the sweep also mis-scoped what
counts.

## Why this is hard to grep for

γ never appears as `5/3` in the affected code. The defect is that γ was **cancelled
analytically** during derivation, leaving an innocuous-looking rational. So the
search target is *literals equal to a function of γ at 5/3*:

| expression | value at γ=5/3 | looks like |
|---|---|---|
| `γ − 1` | 2/3 | `0.6667` |
| `1/(γ−1)` | 3/2 | `1.5` |
| `γ/(γ−1)` | 5/2 | `2.5` |
| `3(γ−1)` | **2** | a bare `2` — invisible |
| `(4π/3)/(γ−1)` | **2π** | `2 * np.pi` |
| `1/(4π(γ−1))` | 3/(8π) | `3` and `8*pi` |

The paper's own Eq. (Rts) shows the same absorption:
`Rts = [3·F_ram·Vb / (8π·Eb)]^{1/2}` is the γ=5/3 form of

    Rts² = F_ram·Vb / (4π(γ−1)·Eb)

## A. CONFIRMED absorbed — the `2π` in the Rahner-A12 equation, **three** copies

| # | file:line | function |
|---|---|---|
| 1 | `bubble_structure/get_bubbleParams.py:130` | `cool_beta_to_Ebdot` |
| 2 | `bubble_structure/get_bubbleParams.py:187` | `Ebdot_to_cool_beta` |
| 3 | `phase1b_energy_implicit/get_betadelta.py:260` | `cool_beta_to_Ebdot_pure` ← **sweep ②'s "pair" missed this one** |

**Proof, from the code's own `bubble_E2P`.** `Eb = (4π/3)·d·Pb/(γ−1)` with
`d = R2³ − R1³`, so the coefficient multiplying `Pb_dot·d²` must be
`(4π/3)/(γ−1)`:

| γ | required | code has |
|---|---:|---:|
| 5/3 | 6.28319 | `2*np.pi` = 6.28319 ✓ |
| 1.4 | 10.47198 | 6.28319 ✗ |
| 1.2 | 20.94395 | 6.28319 ✗ |

At γ=1.4 the ratio is **1.66667** — the same 5/3 factor as the original
R1-vs-`bubble_E2P` disagreement.

**The three implementations agree with each other exactly** (worst rel `0.000e+00`
over 20 000 random states), so this is *not* copy-paste divergence — it is one
defect replicated verbatim. All three must move together.

## B. UNRESOLVED — `1.5` and `0.75` in the same equation

    a_coeff = 1.5  * pdotdot_total / pdot_total
    c_coeff = 0.75 * pdot_total * R1

at all three sites (`get_bubbleParams.py:123-124`, `:177-178`,
`get_betadelta.py:251-252`).

These sit inside the same A12 equation as the confirmed `2π` and derive from the
same `Eb ↔ Pb ↔ Rts` relations, whose γ-dependence is explicit:
`Rts² = F_ram·d / (3(γ−1)·Eb)`. So they are **likely** γ-bearing — but the A12
derivation has **not** been reproduced here, and asserting them without it is
precisely the plausible-but-wrong pattern this audit exists to catch.

> ⚠️ **A partial fix is potentially worse than none.** Changing the `2π` while
> leaving `1.5`/`0.75` frozen would produce a *differently* inconsistent equation.
> Settle the whole A12 derivation at general γ before touching any of it — the
> paper's author can supply the general-γ form directly.

## C. CLEARED — literals that look like γ and are not. **Do not "fix" these.**

| site | literal | what it actually is |
|---|---|---|
| `get_bubbleParams.py:308` | `2 * np.pi` | `pRam = Lmech/(2π r² v_mech)`. **Same literal, unrelated origin** — the 2 cancels one of the 4 in 4π. The audit already REFUTED a proposed fix here (`S6-R-02`); applying it would **double the ram pressure**. |
| `bubble_luminosity.py:401` | `25/4` | Spitzer conduction constant, from κ ∝ T^(5/2) |
| `bubble_luminosity.py:402,404` | `**(5/2)`, `**(2/5)` | conduction temperature profile |
| `bubble_luminosity.py:408` | `-2/5` | `dT/dr` of the same profile |
| `bubble_luminosity.py:442-447` | `2.5 *` | d/dT of T^(5/2) → 5/2. **Not** `γ/(γ−1)`, which is also 5/2 at γ=5/3 — a coincidence that would corrupt the conduction solve if "fixed". |

## D. Verified CORRECT — γ is honoured

| site | expression |
|---|---|
| `get_bubbleParams.py:239` (`bubble_E2P`) | `Pb = (gamma-1)*Eb/V` |
| `get_bubbleParams.py:284` (`get_leak_luminosity`) | `gamma/(gamma-1) * ...` |
| `_functions/operations.py` (`get_soundspeed`) | `sqrt(gamma * k_B * T / mu)` |
| `get_bubbleParams.py:408` + `solve_R1` | **fixed 2026-08-08**, commit `425b9f1` |

## A near-miss worth recording

The round-trip `beta → Ebdot → beta` through the stated inverse pair has a
**worst-case relative error of 2.9e+02** over 20 000 random states. Quoted alone
that looks like a serious defect. It is not:

| percentile | rel err |
|---|---:|
| p50 | 4.8e-15 |
| p90 | 5.4e-08 |
| p99 | 2.5e-03 |
| p100 | 2.9e+02 |

The median is machine epsilon — the inversion is algebraically exact. The tail is
catastrophic cancellation on states drawn independently across 7+ decades, which do
not occur physically. **Not a finding.** Recorded because reporting the worst case
would have been exactly the confident-nonsense failure mode the method is built to
prevent.

## Repro

```bash
# candidate literals, filtered from the Phase-0 ledger
python - <<'PY'
import csv
FILES = ('get_bubbleParams','bubble_luminosity','get_InitPhaseParam','get_betadelta',
         'energy_phase_ODEs','run_transition_phase','run_momentum_phase','operations')
CAND = {'1.5','0.75','2.5','2/3','3/2','5/2'}
for r in csv.DictReader(open('docs/dev/code-audit/data/claims_literals.csv')):
    if any(f in r['file'] for f in FILES) and r['value'] in CAND:
        print(r['file'], r['line'], r['value'])
PY
```
