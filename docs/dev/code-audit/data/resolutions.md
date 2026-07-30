# Orchestrator resolutions of reconciler-flagged decidable questions

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

**Status (2026-07-30):** 🔵 ACTIVE — running log of candidates the orchestrator closed by direct lookup.

## What this file is

Phase-2 reconcilers are forbidden the source: they see only the three lens
reports for their slice. When a reconciler cannot decide a question but *can*
name the exact fact that would decide it, that is a well-posed task for the
orchestrator, who is not blind. This file records those resolutions.

Each entry states the reconciler's open question, the lookup, and the verdict —
including verdicts that **clear** the code, which are the point as much as the
defects are. Nothing here has been through the Phase-5 skeptic gate; a
resolution is evidence, not a ruling.

---

## S9-R-01 — cooling density factor (CIE branch) → **CLEARED**

**Open question.** The S9 reconciler found the CIE branch applies `chi_e · ndens²`
and computed the error factor as `chi_e·(ndens/n_H)²/1.20`: **1.00** if `ndens`
is n_H and `chi_e` = 1.2; **0.833** if `chi_e` = 1.0; **4.41** if `ndens` is
n_tot with `chi_e` = 1.0; **5.29** if n_tot with `chi_e` = 1.2. Lens A had
asserted `ndens` is *total* number density, but flagged it as an inference — A
never saw a call site — and Lens C's spec reading contradicted it. The
reconciler declined to guess and named the deciding facts.

**Lookup.**

- `trinity/_input/registry.py:415` — `chi_e`, default **1.2**, defined as
  *"Electron-per-hydrogen-nucleus factor n_e/n_H = 1 + Z_He*x_He for the HOT
  bubble… Multiplies n_H^2 in the bubble CIE cooling."*
- `trinity/_input/read_param.py:314` — `_chi_e = 1 + _ZHe * _xHe` with the
  inline gloss *"electrons per H nucleus, n_e/n_H"*; at the defaults
  `x_He = 0.1`, `Z_He = 2` this is 1.2.
- Call site `trinity/bubble_structure/bubble_luminosity.py:430`, with `ndens`
  built one line earlier at `:428`:
  `ndens = Pb / ((mu_convert/mu_ion) · k_B · T)`.
- `mu_convert` = μ_H = 1 + 4x_He (mass per **hydrogen nucleus**);
  `mu_ion` = μ_H/(2 + x_He(1+Z_He)) (mean mass **per particle**), both from
  `read_param.py:311-314`. So `mu_convert/mu_ion = 2 + x_He(1+Z_He) = 2.3`,
  which is particles-per-H = n_tot/n_H.
- Ideal gas gives `Pb = n_tot k_B T`, hence
  `ndens = n_tot / (n_tot/n_H) = n_H`.
- Corroborated by `registry.py:366`: *"All densities n in TRINITY are
  hydrogen-nuclei densities n_H."*

**Verdict — the code is correct.** `ndens` is n_H, `chi_e` is n_e/n_H, so the
CIE branch computes `chi_e · n_H² · Λ = n_e n_H Λ` — the error-factor-**1.00**
case, correct for a Λ normalised per `n_e n_H` (the standard CIE convention).

**Lens A's inference was wrong**, and Lens C's headline 4.41× does not apply to
this branch. Recording this as a lens error is deliberate: the reconciler's
refusal to promote an unverified inference to a finding is what prevented a
false S1.

---

## S9-R-02 — non-CIE cube normalisation → **CLEARED (code); doc defect stands**

**Open question.** Lens B's transcription has `trinity/cooling/non_CIE/read_cloudy.py:23`
describing the cube as `[erg cm3 / s]` — i.e. a Λ — while the arithmetic Lens A
transcribed requires `erg cm⁻³ s⁻¹` (volumetric), because the non-CIE branch
applies **no** density factor at all. If the docstring were right, that branch
would be short by n², an error the reconciler bounded at up to 10⁸. The
reconciler's proposed test: scan the `ndens` column of a bundled
`opiate_cooling_*.dat` at fixed (T, Φ).

**Measurement.** Ran on `lib/default/opiate/opiate_cooling_rot_Z1.00_age2.00e+06.dat`
(columns `ID ndens temp phi nedens heat cool`), holding T = 3.16e4 K, Φ = 1e10
fixed and sweeping the 31 densities in that block:

| n_H | cool | cool/n² |
|---|---|---|
| 1e-3 | 3.113e-29 | 3.113e-23 |
| 1e-1 | 6.347e-25 | 6.347e-23 |
| 1e+1 | 4.936e-21 | 4.936e-23 |
| 1e+3 | 7.922e-17 | 7.922e-23 |
| 1e+5 | 8.555e-13 | 8.555e-23 |
| 1e+7 | 8.015e-09 | 8.015e-23 |
| 1e+9 | 7.551e-05 | 7.551e-23 |
| 1e+11 | 7.131e-01 | 7.131e-23 |

`d(log cool)/d(log n) = **2.014**` over 14 decades, and `cool/n²` varies only
3.1e-23 → 8.6e-23 (factor 2.7, consistent with the ionisation state shifting
with density). A Λ would give slope 0.

**Verdict — the cube is volumetric** (`erg cm⁻³ s⁻¹`, n² already included), so
the non-CIE branch applying no density factor is **correct**.

The **docstring is wrong**: `read_cloudy.py:23` labels a volumetric rate as
`[erg cm3 / s]`. That is an **S3** documentation defect, not the S1 the unit
mismatch would otherwise imply. It is exactly the failure mode this audit was
built around — a comment that would make a correct line look wrong (or, read the
other way, license a future "fix" that multiplies by n² and breaks it).

**Reproduce:**

```bash
python3 -c "
import numpy as np
d=np.loadtxt('lib/default/opiate/opiate_cooling_rot_Z1.00_age2.00e+06.dat',skiprows=1)
n,T,phi,cool = d[:,1],d[:,2],d[:,3],d[:,6]
Ts,ps=np.unique(T),np.unique(phi); m=(T==Ts[10])&(phi==ps[10])
print(np.polyfit(np.log10(n[m]),np.log10(cool[m]),1)[0])   # -> 2.014
"
```

---

## Still open in S9 (not resolved here)

- **S9-R-08** — `np.save` of the ragged cube list (`read_cloudy.py:265`) raises
  on the pinned numpy, so no cooling table can be rebuilt from source today.
  Lens A verified this by repro; it needs a fix decision, not a lookup.
- **S9-R-05** — whether the two cooling models actually agree at `Tcut_nCIE`.
  The reconciler corrected Lens C's proposed test (sweeping T across the seam
  returns ≈1 by construction — a false negative); the valid detector compares
  the two models **at the same T**. Not yet run.
