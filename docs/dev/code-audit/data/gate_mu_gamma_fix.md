# Equivalence gate — derived-`mu_*` refusal and `gamma_adia` in the R1 solve

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

**Status (2026-08-08):** 🔵 ACTIVE — gate result for commit `425b9f1`
(`S12a-R-01` `mu_*`, `SIGN-01` `gamma_adia` half). **Result: PASS, byte-identical.**

## The bar, set before editing

The maintainer's requirement was that results stay **identical**. For a change
claimed to be free, CLAUDE.md rule 5 wants the strong form: a value-diff *and* a
byte-identical `dictionary.jsonl`. Unlike the event-dispatch gate, **no diff was
permitted at all**.

## Result — baseline `ba8a62f`, three configs, separate processes

| config | baseline sha256 | post-fix sha256 | verdict |
|---|---|---|---|
| `probe_iscollapse_maxr` | `a55e076fcc339537` | `a55e076fcc339537` | **BYTE-IDENTICAL** |
| `phase6_momentum` | `d2d8ccca44e779d9` | `d2d8ccca44e779d9` | **BYTE-IDENTICAL** |
| `phase6_cfgB` | `6df9c27b29684fc3` | `6df9c27b29684fc3` | **BYTE-IDENTICAL** |

Coverage: 464 snapshots across all four phases (energy, implicit, transition,
momentum).

## Why each half is identical — two different mechanisms

**`mu_*` — identical by construction.** The change adds no arithmetic. It refuses
an input that `read_param.py` was already discarding. Every config that does not
set `mu_*` runs the same code path.

*Sweep safety, checked rather than assumed:* `generate_param_file` emits only
user-authored keys. Against `param/sweep_example.param` it produced
`model_name, path2output, dens_profile, densPL_alpha, nISM, stop_at_rCloud_nSnap,
mCloud, sfe, nCore` — no `mu_*`, so no generated `.param` can trip the guard.

**`gamma_adia` — identical by numerical proof.** Ram-pressure balance gives

    R1² = 2·Lmech·(R2³ − R1³) / (3(γ−1)·v_mech·Eb)

and `3(γ−1)` is **exactly 2.0** at γ=5/3, cancelling the leading 2. The index had
been removed *analytically at 5/3*, which is why grepping the Weaver chain for
`5/3` finds nothing.

**The obvious implementation fails.** Carrying γ explicitly inside `get_r1`:

| form | states differing (of 200 000) | worst rel |
|---|---:|---:|
| `sqrt(2L(r2³−r1³)/(3(γ−1)vEb))` | **13 603** | 4.6e-13 |
| `sqrt(L/v/Eb·(r2³−r1³)/gfac)` | 0 | — |
| **`v` folded: `sqrt(L/(v·gfac)/Eb·(r2³−r1³))`** | **0** | — |

Float multiplication is not associative, so re-ordering alone breaks bit-identity.
The shipped form folds `gamma_factor = 3(γ−1)/2` into the effective velocity
**once per `solve_R1`**, leaving `get_r1`'s body untouched.

## Efficiency — why the folding, and one thing left on the table

`get_r1` runs inside every `brentq` iteration, so the placement matters:

| variant | ns/call | note |
|---|---:|---|
| `get_r1` body today | 931.6 | unchanged by this fix |
| with a γ divide inside | 947.6 | **+16 ns, +1.7 % per iteration** — rejected |
| folding γ into `v` in `solve_R1` | 931.6 | **zero cost**, done once per solve |

**Left deliberately undone:** `np.sqrt` on a Python scalar is **676.8 ns** of that
931.6 ns — **73 % of the function** — against **46.1 ns** for `math.sqrt`. That is
a ~68 % cut to a hot function, and **39× larger than this entire γ change**.

It is *not* a drop-in: `np.sqrt(-x)` returns NaN whereas `math.sqrt(-x)` **raises**,
and `solve_R1`'s own docstring documents depending on the NaN path
(`docs/dev/failed-large-clouds`). It needs its own equivalence gate.

## ⚠️ `gamma_adia` is NOT finished

Step 0 of the plan verified sweep ②'s previously-unverified claim about the
"Rahner A12 pair". The literal `2*np.pi` in `cool_beta_to_Ebdot`
(`get_bubbleParams.py:128`) and `Ebdot_to_cool_beta` (`:184`) **is**
`(4π/3)/(γ−1)` evaluated at γ=5/3:

| γ | `(4π/3)/(γ−1)` required | code has |
|---|---:|---:|
| 5/3 | 6.28319 | `2*np.pi` = 6.28319 ✓ |
| 1.4 | 10.47198 | 6.28319 ✗ |
| 1.2 | 20.94395 | 6.28319 ✗ |

The ratio at γ=1.4 is **1.66667** — the same 5/3 factor as the original
R1-vs-`bubble_E2P` disagreement. This follows from the code's own `bubble_E2P`:
`Eb = (4π/3)·d·Pb/(γ−1)`, which at 5/3 is exactly `Eb = 2π·Pb·d`.

**So γ≠5/3 remains untrustworthy.** `solve_R1` honours it now; the A12 pair does
not. The state is strictly better than before (two inconsistent sites → one) but
it is **not correct**, and must not be described as fixed. Not addressed here at
the maintainer's instruction to report before fixing further.

`get_soundspeed` (`_functions/operations.py`) was also checked and **correctly
honours γ** — no defect there.

## Repro

```bash
python run.py docs/dev/code-audit/harness/probe_iscollapse_maxr.param
python run.py docs/dev/code-audit/harness/phase6_momentum.param
python run.py docs/dev/code-audit/harness/phase6_cfgB.param
sha256sum outputs/*/dictionary.jsonl     # compare against the table above
```
