# Phase-6: momentum-phase asymptotics — the first run to reach phase 2

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

**Status (2026-08-08):** 🔵 ACTIVE — closes the Phase-6 open item "momentum-phase
asymptotics — no run here reached phase 2".

## Provenance

```
config : docs/dev/code-audit/harness/phase6_momentum.param
         (mCloud 1e5, sfe 0.3, stop_t 0.5, transition_trigger blowout)
run    : python run.py docs/dev/code-audit/harness/phase6_momentum.param
fit    : python docs/dev/code-audit/harness/phase6_asymptotics.py \
             outputs/phase6_momentum/dictionary.jsonl
```

**No harness change was needed** — `phase6_asymptotics.py` already carried
`"momentum": {"R2": 1/2, "v2": -1/2}` in its `EXPECTED` table. The probe was blocked
only by the absence of a run that reached phase 2. `transition_trigger blowout`
(`registry.py:408`, a documented alternative to the default `cooling_balance`) ends
the energy phase at `R2 > rCloud` and gets there at `t ≈ 0.09` Myr.

Result: **195 snapshots — energy 97, implicit 50, transition 30, momentum 18.**

## The fit

| phase | qty | measured | expected | delta | rms [dex] | n (late half) |
|---|---|---:|---:|---:|---:|---:|
| energy | R2 | +0.563 | +0.600 | −0.037 | 0.0018 | 49 |
| energy | v2 | −0.429 | −0.400 | −0.029 | 0.0014 | 49 |
| implicit | R2 | +0.546 | +0.600 | −0.054 | 0.0015 | 25 |
| implicit | v2 | −0.509 | −0.400 | −0.109 | 0.0048 | 25 |
| transition | R2 | +0.555 | — | — | 0.0000 | 15 |
| transition | v2 | −0.596 | — | — | 0.0000 | 15 |
| **momentum** | **R2** | **+0.542** | **+0.500** | **+0.042** | **0.0024** | 9 |
| momentum | v2 | −0.217 | −0.500 | +0.283 | 0.0193 | 9 |

Momentum-phase window: `t = 0.124 → 0.500` Myr (0.60 dex), `R2 = 2.06 → 4.39` pc
(0.33 dex).

## Reading it — rms first, as §3 requires

**`R2` in the momentum phase is on the attractor.** rms **0.0024 dex** is the same
order as the energy phase's 0.0018, over a 0.60-dex window. The measured **+0.542**
against the momentum-driven snowplow ideal **+0.500** is a real measurement, and it
is the counterpart to §3's headline: §3 showed the code spans both *energy-driven*
analytic limits; this shows **the momentum-driven `R ~ t^(1/2)` limit is also
recovered**.

**`v2` in the momentum phase is not.** rms **0.0193 dex** is 8× `R2`'s, so by the
rule §3 established — *read the rms column before the exponent* — the **−0.217 must
not be quoted as an attractor measurement**. It is also internally inconsistent with
the `R2` fit: `R ~ t^0.542` implies `v = dR/dt ~ t^−0.458`, not `t^−0.217`. The `R2`
fit is the trustworthy one; the `v2` discrepancy is recorded, not explained.

## A secondary result: this run *tests* §3's explanation

§3 measured the implicit phase at `R2 +1.094, rms 0.0897` and explained the failure
as physics rather than a defect: *"the shell leaves the cloud during it, so the
swept-mass law changes and self-similarity is broken by construction."*

That was an explanation, not a test. This run supplies the test. Because
`transition_trigger blowout` **ends the implicit phase at the cloud boundary**, its
implicit phase never leaves the cloud — and it comes back at **+0.546 with rms
0.0015**, a clean near-Weaver power law instead of 0.0897 of scatter.

**Same code, same physics, implicit phase confined to the cloud ⇒ self-similarity
restored.** That converts §3's explanation from an assertion into a discriminated
one, and it is the kind of check the audit's own trap note ("agreement between
lenses is not verification") calls for.

## Open observation — not a finding

The **transition** phase returns `rms 0.0000` for both `R2` and `v2` over 15 points.
An exactly log-linear trajectory to four decimal places is not what a numerically
integrated segment normally produces, and it may indicate the transition phase emits
analytically-generated or duplicated rows rather than independent integration
output. **Not investigated** — recorded here so a future visit either explains it or
raises it.
