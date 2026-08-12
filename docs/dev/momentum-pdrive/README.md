# momentum-pdrive — is `P_drive = P_HII + P_ram` double-counting one pressure?

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

**Status (2026-08-08):** 🔵 actionable — **measured and open.** The momentum-phase `P_HII` equals the
wind ram pressure to ≤3.6e-16 on every row of every run checked, so the ODE's
`P_drive = P_HII + P_ram` evaluates to exactly `2 × P_ram`. The *measurement* is settled; whether
summing them is correct is a question about model intent and is **not resolved here**. Nothing has
been changed in `trinity/`.

---

## 1. What was measured

Found 2026-08-08 while computing the shell force budget for an unrelated question (how much of
TRINITY's shell driving is non-wind, in the context of comparing against the wind-only Lancaster+2021
simulations — `transition/kappa-3way/FINDINGS §16e`).

Harness: `harness/check_phii_pram.py` · artifact: `data/phii_pram_evidence.csv`

```
python docs/dev/momentum-pdrive/harness/check_phii_pram.py \
    outputs/bench5/bench1_m5e4_r20__none_diag \
    outputs/bench5/bench2_m1e5_r10__none_diag \
    outputs/bench5/bench3_m1e5_r5__none_diag
```

| run | momentum rows | `P_ram` dynamic range | max relΔ `P_HII` vs `P_ram` | `P_drive == 2·P_ram` |
|---|---|---|---|---|
| bench1_m5e4_r20 | 30 | 1.9× | 2.8e-16 | all rows |
| bench2_m1e5_r10 | 95 | 22.3× | 3.4e-16 | all rows |
| bench3_m1e5_r5 | 104 | 88.5× | 3.6e-16 | all rows |

The recorded `P_HII` also matches `pRam = L/(2πR₂²v)` **recomputed from each row's own
`Lmech_total`, `v_mech_total`, `R2`** to the same 3.6e-16. So this is not an artifact of how the two
are stored — the quantity written as the photoionized-gas pressure *is* the wind ram pressure.

Consequently `F_HII == F_ram` on every row, and the reported force budget carries no independent
photoionized-gas contribution.

## 2. Why it is not a simple aliasing bug

The two are equal to ~1–2 ULP but **not bit-identical** (bit-equal on only 6/30, 35/95, 38/104 rows;
the rest differ in the last bit). A plain `P_HII = P_ram` assignment would be bit-identical
everywhere. So `P_HII` is being *computed*, by a path that is algebraically equivalent to `pRam`.

The likely mechanism — ⚠️ **inferred from a code comment, not measured**:
`trinity/shell_structure/shell_structure.py:239-251` caps

```python
n_IF_Str = min(n_IF_Str, shell_n0)      # "Cap: n_IF_Str ≤ shell_n0 (pressure equilibrium for thin skins)"
```

If that cap binds, the ionized skin's density is pinned to the shell's inner density, which is itself
set by balance against the driving pressure. Then
`P_HII = (mu_convert/mu_ion_shell)·n_IF_Str·k_B·TShell_ion` reproduces the confining pressure up to
rounding — exactly what is observed. On this reading the equality is a **consequence of the physics
the cap encodes**, not a coding slip.

## 3. Why that makes the sum the real question

The ODE right-hand side (`trinity/phase2_momentum/run_momentum_phase.py:445`, and the same pair at
`:265` for the pre-ODE diagnostic path):

```python
P_drive    = snapshot.P_HII + P_ram
F_pressure = FOUR_PI * R2**2 * (P_drive - P_ext)
```

**This is in the integrator, not the diagnostics** — it sets `dv₂/dt`, so it propagates into R₂(t),
the fate, and the stopping outcome of every run that reaches the momentum phase.

If §2 is right and the ionized skin is in pressure equilibrium with the wind, then the skin
*transmits* the confining pressure to the neutral shell; it does not supply a second, independent
push. Adding `P_HII + P_ram` would then count the same physical pressure twice and drive the shell at
2× the justified force.

**The counter-case** — and it is why this doc does not call it a bug: if the model intends the
photoionized region to be a genuinely separate reservoir acting alongside the wind (rather than a thin
equilibrium skin), the sum is correct and the near-equality of the two terms is a coincidence of this
parameter regime that would break elsewhere. Deciding between these needs the model's intent, which
lives with the maintainer, not in the code.

## 4. What is NOT affected

- **The kappa-3way campaign's Θ measurements.** Θ_cum is integrated over the *implicit* phase, which
  ends at the transition; the momentum phase begins after. `FINDINGS §14`'s re-baseline and `§16`'s
  Eq-10 screen both read implicit-phase rows only.
- ⚠️ **But fates and stopping outcomes are downstream of the momentum phase**, so anything that quotes
  a fate, a collapse/dissolution time, or a final radius — including K3's fate-determinism arm — is
  potentially affected. Not audited.

## 5. Open questions, in order

1. **Is the sum intended?** (maintainer — §3). Everything else waits on this.
2. If it is a double count: what is the correct `P_drive` — `max(P_HII, P_ram)`, `P_ram` alone, or a
   proper two-zone treatment? And does the same pairing appear in
   `phase1c_transition/run_transition_phase.py`, which has the structurally identical
   `P_HII`/`P_ram` block?
3. What changes in the benches if it is fixed? A/B on the three arms above is ~30 min.
4. Does the `n_IF_Str ≤ shell_n0` cap bind in *all* regimes, or only these? If it unbinds somewhere,
   `P_HII` and `P_ram` would separate and the double count would change size with regime — which
   would make it a regime-dependent bias rather than a clean factor of 2.
