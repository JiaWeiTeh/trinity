# phii-identity — `P_HII` is the confining pressure relabelled, in every phase

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

**Status (2026-08-12):** 🔵 actionable — **evidence gathered and the mechanism is now proved,
not inferred.** Five workstreams across three unmerged branches each measured `P_HII` equal to the
local confining pressure to 4–10 digits. This doc consolidates them and shows the equality is an
**exact algebraic identity** — `P_HII` re-derives its own input — whenever the `n_IF_Str ≤ shell_n0`
cap binds. **Nothing in `trinity/` has been changed.** What to *do* about it is an intent question
for the maintainer (§7). **Fix effort:** planned and pre-registered in
[`PLAN.md`](PLAN.md) (branch `bugfix/phii-pt1`) — candidates, config matrix, batch gates, and
the running ledger all live there; this README stays the evidence record.

---

## 1. The finding in one paragraph

Three branches reported the same number from different directions:
`feature/low-winds-regime` saw `Pb/P_HII = 1.0000000000`, `hotfix/other-magic-numbers` saw
`P_HII/Pb = 1.0000`, and `feature/threeway-pt2` saw `P_HII/P_ram = 1` to ~1.6 ULP. These are not
three findings but **one**, seen in three phases. Whenever the Strömgren density hits its cap,
`P_HII` is computed by inverting, term for term, the algebra that produced the cap — so it returns
the confining pressure it was capped against. The ionizing photon rate `Qi`, the escape fraction,
the ionized volume: none of them survive. `P_HII` is a relabelling, and the "photoionized gas
pressure" channel carries no independent physics in that regime.

## 2. The mechanism — algebra, not coincidence

Verified against current source on `main` @ `731ac50` (all three lines still present as cited):

| step | source | expression |
|---|---|---|
| 1 | `trinity/shell_structure/shell_structure.py:124-126` | `shell_n0 = (mu_ion_shell / mu_convert / (k_B · TShell_ion)) · Pb` |
| 2 | `trinity/shell_structure/shell_structure.py:251` | `n_IF_Str = min(n_IF_Str, shell_n0)` |
| 3 | `run_energy_phase.py:224` · `run_transition_phase.py:564` · `run_momentum_phase.py:634` | `P_HII = (mu_convert / mu_ion_shell) · n_IF_Str · k_B · TShell_ion` |

Step 1 defines the shell's inner density by **pressure balance against `Pb`**. Step 2 caps the
Strömgren density at it ("pressure equilibrium for thin skins"). Step 3 converts a density back
to a pressure using **the same three factors** — so when the cap binds, substituting (1) into (3):

```
P_HII = (mu_c/mu_i) · [(mu_i/mu_c) / (k_B·T) · Pb] · k_B · T  ≡  Pb
```

Every factor cancels. This is an identity in exact arithmetic, **independent of regime, cloud,
mass, or feedback strength** — it cannot "come out differently" anywhere the cap binds.

**Why the momentum phase reports `P_ram` instead of `Pb`:** it is the same identity.
`run_momentum_phase.py:585` sets `params['Pb'].value = pRam(R2, Lmech_total, v_mech_total)` — with
the explicit comment *"Set Pb to ram pressure so shell inner-edge density is physically
meaningful"*. So `Pb` **is** the ram pressure there, `shell_n0` is built from it, and `P_HII` gives
it back. One mechanism, two labels. (`momentum-pdrive/README.md` §2 called this mechanism
"inferred from a code comment, not measured" — it is now measured, and the `Pb := pRam` line is the
missing link it did not cite.)

## 3. The evidence, gathered

Full table with per-row provenance: `data/phii_identity_evidence.csv`. Source artifacts live on
the branches named; **none of them are on `main` @ `731ac50`** except the `html-insights` row.

| workstream | branch @ SHA | phase | ratio | value |
|---|---|---|---|---|
| `weak-winds` | `feature/low-winds-regime` @ `ee84fc7` | implicit | `Pb/P_HII` | **1.0000000000** (t = 0.016 and 0.295 Myr) |
| `weak-winds` | same | energy / late | `Pb/P_HII` | 0.3333 → 0.9781 → … → 1.0069 (**cap slack at both ends**) |
| `switchon-successor` | `hotfix/other-magic-numbers` @ `704c96b` | energy | `P_HII/Pb` | **1.0000** exactly, first 6 snapshots of `simple_cluster` |
| `phase1a-init` | same | energy (1a) | `P_HII/Pb` | **1.0** to all printed digits, all 128 snapshots, M43 probe |
| `transition/cleanroom` | same | implicit | `Pb ≡ P_HII` | machine precision, **all 6 configs** |
| `html-insights` verification #5 | `main` @ `731ac50` | implicit | `Pb ≡ P_HII` | machine precision ✅-verified |
| `momentum-pdrive` | `feature/threeway-pt2` @ `96707dc` | momentum | `P_HII/P_ram` | **1.0** to 2.8e-16 / 3.4e-16 / 3.6e-16 over 30 / 95 / 104 rows |

The `momentum-pdrive` arms are the strongest single piece of evidence, because they hold the
identity across a **88× dynamic range in `P_ram` within one run** — a coincidence cannot track a
quantity over two orders of magnitude.

The `weak-winds` row is the most informative about **scope**: the ratio departs from 1 at both
ends of the run (0.333 at t=0, 1.0069 at 15 Myr). So the cap is *not* always binding — `P_HII` is
genuinely independent in early phase 1a and at late times, and the identity holds over the long
middle. Any fix must handle both sides of that transition.

## 4. Why it is not bit-identical — and why that is reassuring, not puzzling

`momentum-pdrive/README.md` §2 flagged that the two are equal to ~1–2 ULP but bit-identical on only
6/30, 35/95 and 38/104 rows, and read that as evidence `P_HII` is *computed* rather than assigned.
That reading is right, and the residual is fully explained by the round trip itself: `mu_ion_shell/
mu_convert` and `mu_convert/mu_ion_shell` are each rounded separately (they are not exact
reciprocals in binary), and the two sides associate `k_B · TShell_ion` differently.

`harness/roundtrip_ulp.py` models exactly those three lines over a 12-dex `Pb` sweep with production
constants (`x_He=0.1`, `Z_He_shell=1`, `TShell_ion=1e4`), in both internal and cgs units:

```
python docs/dev/phii-identity/harness/roundtrip_ulp.py     # ~1 s, no simulation needed
```

| | max relΔ | in ULP | bit-equal fraction |
|---|---|---|---|
| model, astro (internal) units | 4.44e-16 | 2.00 | 0.289 |
| model, cgs units | 4.44e-16 | 2.00 | 0.327 |
| **measured in production** (3 benches) | 2.8e-16 – 3.6e-16 | 1.27 – 1.64 | 0.200 / 0.368 / 0.365 |

The arithmetic model predicts both the ULP ceiling and the ~30% bit-equal rate that production
shows. Artifact: `data/roundtrip_ulp.csv`. **Conclusion: the residual is float rounding on an exact
identity — there is no third quantity hiding in `P_HII`.**

## 5. Where it lands in the ODEs — the momentum phase is the outlier

This is the part no single branch had, because each looked at its own phase. Every `P_drive` site
on `main` @ `731ac50`:

| phase | source | `P_drive` | effect when `P_HII ≡ P_confining` |
|---|---|---|---|
| energy (1a) / implicit (1b) | `phase1_energy/energy_phase_ODEs.py:256, 388` (the `else` branch) | `max(Pb, P_HII)` | **absorbed** — `max(Pb, Pb) = Pb`, exact no-op |
| implicit (1b) | `phase1b_energy_implicit/run_energy_implicit_phase.py:532` | `max(Pb, P_HII)` | **absorbed** — exact no-op |
| **transition (1c)** | `energy_phase_ODEs.py:253, 385` (gated `current_phase == 'transition'`) · `phase1c_transition/run_transition_phase.py:331` | `max(Pb, P_HII + P_ram)` | **not absorbed** — see below |
| **momentum (2)** | `phase2_momentum/run_momentum_phase.py:265, 445` | **`P_HII + P_ram`** — bare sum, no `max` | **`= 2 · P_ram`.** Shell driven at twice the justified pressure |

**The `max()` only protects phases 1a/1b.** There, `max(Pb, Pb)` is exactly `Pb`, so the identity is
invisible in the trajectory — which is precisely why `_analysis/check_yesno.py` exists and why
toggling `include_PHII` changes nothing there (its docstring already states
`"P_drive=max(Pb,P_HII)=Pb identically"`).

**In the transition phase the `max` looks like a guard but never binds.** With `P_HII ≡ Pb` and
`P_ram > 0` (it is non-zero only in transition — `energy_phase_ODEs.py:392`),

```
max(Pb, P_HII + P_ram)  =  max(Pb, Pb + P_ram)  =  Pb + P_ram      — always, for any P_ram > 0
```

so the second argument wins on **every** step and the double count lands in full. This sharpens
`momentum-pdrive`'s open question 2 ("does the same pairing appear in `phase1c_transition`?"): yes —
and the `max` wrapper does **not** make the exposure conditional, as one might assume from reading
it. Phases 1c and 2 are both affected; only 1a/1b are safe.

Both sites are in ODE right-hand sides (`vd` at `energy_phase_ODEs.py:263`; `F_pressure` at
`run_momentum_phase.py:448`), not diagnostics — so this propagates into `R2(t)`, the force budget,
the fate, and the stopping outcome of every run that reaches transition or momentum.

## 6. Where the branches agree, and where they read it differently

- **All three agree on the measurement and on the mechanism** (the `n_IF_Str ≤ shell_n0` cap).
- **`weak-winds` draws the physics conclusion:** the study's H2 ("dense clouds are HII-supported and
  so wind-insensitive") is *mechanically wrong* in the cap-limited regime — weaken the wind, `Pb`
  falls, and `P_HII` falls with it. It correctly notes its own phase reads
  `P_drive = max(Pb, P_HII)`, "so the terms compete rather than sum."
- **`phase1a-init` draws the numerics conclusion:** the identity interacts with per-segment
  snapshot freezing to make a **stale-pressure ratchet** — `max(Pb_live, P_HII_frozen)` with
  `P_HII == Pb` means a segment's driving pressure can never fall below its segment-start value.
  Mild at GMC scale, catastrophic at compact-probe scale (`Pb` falls ~7 dex within one segment).
- **`momentum-pdrive` draws the force-budget conclusion:** `F_HII == F_ram`, so the reported budget
  carries no independent photoionized contribution, and the ODE drives on `2 · P_ram`.

They are consistent; each is a different downstream consequence of §2.

## 7. Open questions — for the maintainer, in order

1. **Is the `P_HII + P_ram` sum intended?** (§5) Everything else waits on this. If the ionized skin
   is a thin equilibrium layer that *transmits* the confining pressure, the sum double-counts in
   **both** transition and momentum. If the model intends a genuinely separate reservoir, the sum
   is right and the near-equality is telling us the *cap* is wrong. **This is a physics-intent
   call, not a code call.** Note that "wrap it in a `max`" is *not* an available fix: transition
   already has one and it never binds (§5).
2. **Should the cap be a cap at all?** Under it, `P_HII` cannot exceed `Pb` *by construction*, so
   the ionized-gas channel can never be the dominant driver anywhere in the code — which is a
   strong physical claim to make implicitly, in a `min()`. *(Update 2026-08-12: maintainer states
   the cap's origin is numerical — a guard against the ΔV→0 blow-up of `n_IF_Str` — not a physics
   claim. See `PLAN.md` §2; the guard-replacement candidate C2b follows from this.)*
3. **Is `include_PHII` doing what its name promises?** It gates `P_HII` in all four phases
   (`run_energy_phase.py:223`, `run_energy_implicit_phase.py:980,1378`,
   `run_transition_phase.py:563,844`, `run_momentum_phase.py:633`), but in phases 1a–1c the `max`
   makes it a no-op whenever the cap binds. Runs labelled `_yesPHII` / `_noPHII` differ only via
   the momentum phase — worth knowing before any published comparison rests on that label.
4. **What changes if it is fixed?** `momentum-pdrive` estimates an A/B on its three benches at
   ~30 min. That is the cheapest next measurement and should precede any code change.
5. **Fates are downstream and unaudited.** Anything quoting a fate, collapse/dissolution time, or
   final radius passes through transition and/or momentum. `kappa-3way`'s Θ measurements are *not*
   affected (implicit phase only, where the `max` genuinely absorbs the identity), but its
   fate-determinism arm may be. Note `weak-winds`' smoke pair reached collapse **via** transition
   and momentum (`energy → implicit → transition 0.160 → momentum 0.182 → collapse`), so its fate
   flip is downstream of both affected sites — the *direction* of its conclusion is unaffected
   (weakening the wind lowers `Pb`, hence `P_HII`, hence both terms), but the quantitative collapse
   time is not a clean number until §7.1 is settled.

## 8. Discrepancies found while gathering (flagged, not fixed)

- **`momentum-pdrive/data/phii_pram_evidence.csv` vs its README.** The CSV reports
  `F_HII_equals_F_ram_all_rows = False` for all three benches, while README §1 states
  "Consequently `F_HII == F_ram` on every row". Both are defensible — the harness tests *bit*
  equality (`check_phii_pram.py`: `if F_HII != F_ram`), which §4 above predicts will fail ~65% of
  the time — but the column name reads as a contradiction of the prose. Suggest renaming the
  column to `F_HII_bitequal_F_ram_all_rows`, or comparing within tolerance.
- **`momentum-pdrive/README.md` §2** labels the mechanism ⚠️ *"inferred from a code comment, not
  measured"*. §2 and §4 here supersede that: it is now derived algebraically and confirmed
  numerically. Worth folding back when that branch next moves.
- **Nothing here contradicts any branch's measurements.** All numbers reproduced as reported.

## Layout

```
README.md                      this file — the consolidated evidence
PLAN.md                        the fix effort: candidates, gates, batch ladder, ledger, dated log
data/phii_identity_evidence.csv  every sighting, one row each, with branch + SHA provenance
data/roundtrip_ulp.csv         the float round-trip model output
harness/roundtrip_ulp.py       reproduces the ULP signature (pure arithmetic, ~1 s)
```
