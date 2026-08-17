# phii-identity — `P_HII` **was** the confining pressure relabelled, in every phase (fixed by C3c, `c43a50e`)

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

**Status (2026-08-14):** 🟡 **the fix (C3c) shipped — this file is now the historical evidence record for behaviour `main` no longer has.** `c43a50e` (PR #738) replaced the capped-Strömgren `P_HII` with a cavity-Strömgren regime switch at all six call sites, so the identity documented below describes the code **before** 2026-08-14. Read it to understand why `F_HII`/`include_PHII` behaved as they did, and `PLAN.md` for what replaced them. **Evidence gathered, mechanism proved, measured directly, and independently audited.** An adversarial audit on 2026-08-13 corrected several claims in this file (§5's "1a/1b are safe", §7.3's `_yesPHII` scope, §8's reproduction breadth) — each is marked in place. `PLAN.md` §9 carries the full list; do not quote a figure from an unmarked earlier revision. Batches 0/1 (see `PLAN.md`) confirm the identity on 100% of implicit,
transition and momentum rows across five configs, show the cap binding on **100% of rows in every
phase** (so §3's cap-slack reading is retracted), and size the double-count at **1.82× median in
transition, exactly 2.000× in momentum**. **Original framing below.** Five workstreams across three unmerged branches each measured `P_HII` equal to the
local confining pressure to 4–10 digits. This doc consolidates them and shows the equality is an
**exact algebraic identity** — `P_HII` re-derives its own input — whenever the `n_IF_Str ≤ shell_n0`
cap binds. ~~**Nothing in `trinity/` has been changed.**~~ ⚠️ **Corrected 2026-08-14:** true when
written, false now — C3c landed in `c43a50e`. What to *do* about it was an intent question for the
maintainer (§7); it has been answered. **Fix effort:** planned and pre-registered in
[`PLAN.md`](PLAN.md) (branch `bugfix/phii-pt1`, merged to `main` 2026-08-14) — candidates, config
matrix, batch gates, and the running ledger all live there; this README stays the evidence record.

---

## 1. The finding in one paragraph

> **STATUS 2026-08-14 — this section describes the DEFECT, which is fixed.** Everything below is
> written in the present tense about *stock* `P_HII` (the capped-Strömgren pressure). That code is
> gone: `c43a50e` (PR #738) replaced it with the C3c confinement regime switch, and `P_HII` is now
> a cavity-Strömgren pressure that returns **exactly 0.0** while the ionised gas is confined. Read
> this section as the historical diagnosis; see `PLAN.md` §3c for what shipped. Measured on a fresh
> stock-vs-C3c pair (`data/b7_regime_trajectory.csv`), the stock identity still reproduces at
> `|P_HII − Pb|/Pb` median **1.3e-16**, so the diagnosis below is correct — it is just no longer
> the behaviour of the code.

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
| 2 | `trinity/shell_structure/shell_structure.py:253` | `n_IF_Str = min(n_IF_Str, shell_n0)` |
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
| `weak-winds` | same | energy / late | `Pb/P_HII` | 0.3333 → 0.9781 → … → 1.0069 (~~cap slack at both ends~~ — **retracted, see below**) |
| `switchon-successor` | `hotfix/other-magic-numbers` @ `704c96b` | energy | `P_HII/Pb` | **1.0000** exactly, first 6 snapshots of `simple_cluster` |
| `phase1a-init` | same | energy (1a) | `P_HII/Pb` | **1.0** to all printed digits, all 128 snapshots, M43 probe |
| `transition/cleanroom` | same | implicit | `Pb ≡ P_HII` | machine precision, **all 6 configs** |
| `html-insights` verification #5 | `main` @ `731ac50` | implicit | `Pb ≡ P_HII` | machine precision ✅-verified |
| `momentum-pdrive` | `feature/threeway-pt2` @ `96707dc` | momentum | `P_HII/P_ram` | **1.0** to 2.8e-16 / 3.4e-16 / 3.6e-16 over 30 / 95 / 104 rows |

The `momentum-pdrive` arms are the strongest single piece of evidence, because they hold the
identity across a **88× dynamic range in `P_ram` within one run** — a coincidence cannot track a
quantity over two orders of magnitude.

The `weak-winds` row *appeared* to be the most informative about **scope**: its ratio departs
from 1 at both ends (0.333 at t=0, 1.0069 at 15 Myr), suggesting the cap is not always binding.

⚠️ **Retracted 2026-08-12 by `PLAN.md` Batch 1 — there is no cap-slack window.** Measuring the cap
directly (`n_IF_Str_raw`, the pre-cap value) shows it binding on **100% of rows in every phase** of
every config tested. weak-winds reconstructed `Pb` as `F_ram/4πR2²`, and `F_ram` carries the
*ramped* bubble pressure (`get_effective_bubble_pressure` pulls `R1 → 0` for the first
`dt_switchon = 1e-3` Myr) while `P_HII` carries the un-ramped one — so the reconstruction is off by
exactly that ramp factor, which is ~3 early and 1 later. Reading `Pb` directly gives
1.0000000000 at t=0. The handful of rows where `P_HII ≠ Pb` are `Pb` staleness at the 1a→1b
handoff, with the cap still bound. A fix therefore does **not** need to handle a slack regime — but
it does need to handle the ramp mismatch (`PLAN.md` §9, "D-ramp").

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

⚠️ **Corrected 2026-08-13 (audit).** The paragraph below says the `max` makes 1a/1b safe. That is
true only in the narrow sense that `P_drive == Pb == P_HII` there. It is **not** a no-op: in phase
1a the ODE compares against the **ramped** bubble pressure (`get_effective_bubble_pressure` pulls
`R1 → 0` for the first `dt_switchon = 1e-3` Myr) while `P_HII` carries the **un-ramped** one, so
inside that window `P_HII` wins the `max` and supplies up to **3.31×** the ODE's own pressure —
measured in `PLAN.md` §1(3) (D-ramp) and visible as `Pdrive_over_Fram_max` = 3.06–3.31 in
`data/b0_identity_grid.csv`. Batch 4a's largest trajectory shifts all landed inside that window,
i.e. in the phase this section calls safe. Read "safe" as **"a genuine no-op in 1b, and in 1a only
outside the `dt_switchon` window."**

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
it. Phases 1c and 2 are both affected. 1a/1b are safe **only in the `max` sense**, and 1a is not safe inside the `dt_switchon` window — see the correction above.

Both sites are in ODE right-hand sides (`vd` at `energy_phase_ODEs.py:263`; `F_pressure` at
`run_momentum_phase.py:448`), not diagnostics — so this propagates into `R2(t)`, the force budget,
the fate, and the stopping outcome of every run that reaches transition or momentum.

**Update 2026-08-16 — the momentum phase is *still* the outlier under the fix, and it is not a
calibration error.** C3c (`c43a50e`) replaced the relabelled `P_HII`, but its successor `P_C3a`
comes out dominant over `P_ram` in the momentum phase of every configuration measured. Batch 8
(`PLAN.md` §Batch 8) tested that magnitude against the classical D-type limit and found the shipped
`get_phii_c3c` reproduces **Hosokawa & Inutsuka (2006) exactly** — 0.0000% deviation over
`R/R_St ∈ [2,50]`, index 0.57124 vs 4/7, sitting `(4/3)^{2/7}` = 8.56% above Spitzer as the
momentum-equation closure requires — with the check demonstrably able to resolve a 0.1% pressure
error. So the remaining momentum-phase question is about **model structure** (`P_C3a ∝ R2^{−3/2}`
vs `P_ram ∝ R2^{−2}`: does a real momentum-phase cavity stay Strömgren-filled?), **not** about
C3a's normalisation. Pinned by `test/test_phii_c3c_spitzer.py`.

**Update 2026-08-17 (Batch 9) — the geometry half is now measured, and it is not the answer
either.** C3a spreads the shell-absorbed photon budget over the whole cavity, `(4/3) pi R2^3`, while
trinity's own shell solve puts the ionised gas in the shell itself. Correcting the volume gives
`n_layer/n_cavity = sqrt(R2/(3 dR))`, and on the B3M momentum rows the shell is **thick**
(`dR/R2` = 0.670–1.308), so the correction **lowers** `P_HII` by 0.51–0.71×.

**How much it lowers it depends on how the layer density is computed, and G9.4 settled that.**
Replaying the shipped `shell_structure_pure` (`harness/layer_density_check.py`) shows the analytic
thin-layer Strömgren scaling **overestimates** the real profile's recombination-equivalent density by
up to **3.17×** — G9.4's 2× bar, so **FALSIFIED**. The cause is measured and exact where the layer is
thin: a Strömgren balance assumes every absorbed ionising photon recombines, but the real shell loses
61–75% of them to **dust**, and `sqrt(recomb/Qi_abs)` reproduces the gap to three decimals (0.497 vs
0.496 in energy; 0.907 vs 0.906 in implicit). On the profile form, momentum `P_HII/P_ram` goes
6.165 → **1.545** (1.322–1.666, *falling* with time) — still HII-dominated on every row, but by ~50%
rather than ~500%.

So both the calibration (Batch 8) and geometry (Batch 9) explanations are excluded, and what is left
is the **pressure coupling** — **D5** in `PLAN.md` §7, a physics-intent question. 🔍 Though note:
an extrapolation from stage 3's `Lw^−0.33` briefly suggested inversion at a physical `Lw ≈ 3.4`.
**Batch 10 tested that on `B3MW3`/`B3MW10` and falsified it.** The profile form does not inherit the
cavity exponent — stronger winds *thin* the shell (`dR_ion/R2 ∝ Lw^−0.3375`), which *raises* the
geometry correction and cancels ~43% of the cavity decline, leaving a net `Lw^−0.1133`. Momentum
medians 1.5451 / 1.3412 / 1.1902 for `Lw` = 1/3/10, so inversion moves only to `Lw` ≈ 46.5 — still
unphysical. **D5 is the route.** One useful by-product: `B3MW10` dips to `dR_ion/R2` = 0.3197, below
the `R2/3` break-even, so the geometry correction's **sign is wind-dependent** — which is why the
Batch 9 scope ("raises `P_HII`") and its verdict ("lowers it") were both partial views. ⛔ The Batch 9 *scope* first claimed the geometry
correction was one-signed and *raised* `P_HII`; that was measured before momentum was covered and is
retracted in `PLAN.md` §Batch 9.

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
   makes it a no-op whenever the cap binds — *except* in phase 1a inside the `dt_switchon` window,
   where `P_HII` carries the un-ramped bubble pressure and therefore does change the answer (D-ramp,
   `PLAN.md` §1(3)), and in transition, where §5 shows the `max` never binds. ⚠️ *Corrected
   2026-08-13 (audit): this section previously said `_yesPHII`/`_noPHII` differ only via the momentum
   phase, which contradicted §5 four sections earlier.* Runs so labelled differ via 1a, 1c and 2 — worth knowing before any published comparison rests on that label.
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
- **`weak-winds` FINDINGS' cap-slack claim.** Its "P_HII is genuinely independent only when the cap
  is slack: early phase 1a (ratio 0.33 → 0.98) and late times (1.0069)" does not survive direct
  measurement (§3). The measurements are right; the `Pb = F_ram/4πR2²` reconstruction behind them
  is not. Worth folding back when that branch next moves.
- **`F_ram` is not `4πR2²·P_drive`.** The reported `F_ram` uses the ramped pressure, so any force
  budget read from snapshots understates the shell-facing force by up to ~3× inside the
  `dt_switchon` window. `params['F_ram']` is never read back by the solver, so this is a reporting
  defect, not a dynamics one — but it is what every force-budget analysis here consumes, and it is
  what produced the item above.
- **All other numbers reproduced as reported.** ⚠️ *Narrowed 2026-08-13 (audit): an earlier revision
  added "including momentum-pdrive's three benches". Only **bench3's** config was re-run, under a
  `stop_t 1.5` override giving **34** momentum rows over a **6.7×** `Pb` range — not the 104 rows
  over 88× that momentum-pdrive reported. The identity itself reproduces (relΔ 2.39e-16), but the
  other two benches were never re-measured.*

## Layout

```
README.md                      this file — the consolidated evidence
PLAN.md                        the fix effort: candidates, gates, batch ladder, ledger, dated log
data/phii_identity_evidence.csv  every sighting, one row each, with branch + SHA provenance
data/roundtrip_ulp.csv         the float round-trip model output
harness/roundtrip_ulp.py       reproduces the ULP signature (pure arithmetic, ~1 s)
```
