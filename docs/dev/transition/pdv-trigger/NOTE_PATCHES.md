# Patches for the Paper-II note — *"Adding unresolved interface cooling to TRINITY without double-counting"*

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
> a committed artifact under `docs/dev/` (a CSV/table in `docs/dev/<workstream>/data/`, or a
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

Four ready-to-paste edits for the maintainer's external `.tex` note (the `.tex` is not in this repo,
so these are drop-in snippets + a precise change list). Every number traces to a committed CSV under
`data/`; reproduce with the builder named in each patch. **Caveat carried through:** the `f_mix`
values are a **frozen-trajectory screen** (`make_fmix_table.py` header) — the LIVE matched-`t` edge
runs now in progress (`runs/`, PLAN.md §Task B) will confirm or refine them; the note should cite
them as a *screen pending live confirmation*, not a final calibration.

---

## Patch 1 — Table 2: use the no-PdV (recommended-trigger) column; headline `f_mix ≈ 1.4–2.8`

**Why.** The note's recommended trigger keeps PdV **out** of the transition criterion (PdV in the ODE
only — the reversible/irreversible argument). But the imported Table 2 headline (`f_mix ≈ 1.1–1.5`)
was computed with **PdV inside** the screening ratio, which understates the boost by ~`PdV/Lmech`. The
consistent column solves `(Lmech − f·Lcool)/Lmech = 0.05` at blowout → `f = 0.95 / (Lcool/Lmech)`.
Source: `data/fmix_table.csv` (builder `data/make_fmix_table.py`, both conventions).

Paste-ready (booktabs):

```latex
\begin{table}
\centering
\caption{Cooling-boost multiplier $f_\mathrm{mix}$ that brings the energy$\to$momentum
trigger to threshold ($\epsilon=0.05$) \emph{at blowout}, per config, density-ordered
(densest first). $f_\mathrm{mix}^{\rm out}$ uses the recommended trigger (PdV in the ODE only,
\emph{out} of the criterion) and is the consistent headline; $f_\mathrm{mix}^{\rm in}$ folds PdV
into the screening ratio and is shown only to expose the $\sim\!P\mathrm{d}V/L_\mathrm{mech}$
understatement. Frozen-trajectory screen; live runs pending.}
\label{tab:fmix}
\begin{tabular}{lcccc}
\hline
Config & $L_\mathrm{cool}/L_\mathrm{mech}$ & $P\mathrm{d}V/L_\mathrm{mech}$ &
  $f_\mathrm{mix}^{\rm in}$ & $f_\mathrm{mix}^{\rm out}$ \\
\hline
small\_dense\_highsfe  & 0.697 & 0.182 & 1.10 & 1.36 \\
simple\_cluster        & 0.667 & 0.206 & 1.12 & 1.42 \\
midrange\_pl0          & 0.610 & 0.219 & 1.20 & 1.56 \\
be\_sphere             & 0.511 & 0.308 & 1.26 & 1.86 \\
pl2\_steep             & 0.342 & 0.441 & 1.49 & 2.78 \\
large\_diffuse\_lowsfe & 0.250 & 0.169 & 3.13 & 3.81 \\
\hline
\end{tabular}
\end{table}
```

**Headline sentence** to accompany it:

> A modest cooling boost of $f_\mathrm{mix}\approx1.4$–$2.8$ fires the energy$\to$momentum handoff
> right at blowout for the five compact/normal clouds; the one diffuse outlier
> (`large_diffuse_lowsfe`) needs $f_\mathrm{mix}\approx3.8$. That spread — a factor $\sim\!2.8$ in the
> required boost across the density grid — is itself the evidence that a \emph{constant} multiplier
> cannot place the transition at blowout everywhere, and motivates the density-coupled
> $\theta_\mathrm{target}(\mathrm{Da})$ / $\kappa_\mathrm{eff}$ closure (note §endgame).

---

## Patch 2 — every inline `f_mix ≈ 1.1–1.5` → `1.4–2.8`

Find/replace each occurrence of the old headline in prose. Suggested replacement text:

> $f_\mathrm{mix}\approx1.4$–$2.8$ (recommended no-PdV trigger; the earlier $1.1$–$1.5$ figure folded
> PdV into the \emph{screening} criterion and so understates the boost by $\sim\!P\mathrm{d}V/L_\mathrm{mech}$).

---

## Patch 3 — soften / properly ground the double-count claim

**Why.** The note asserted a $5\times10^5$-draw Monte-Carlo result as the *proof* of no double-counting,
with no committed script. The primary guarantee is actually **algebraic** (single-count *by
construction*); the MC only **corroborates**. Now backed: `data/doublecount_mc.csv`
($5\times10^5$ draws, seed 20260624: **0** enter the $2\theta$ region, max counted/single $=1.0$);
builder `data/make_doublecount_mc.py`.

Replace the bare assertion with:

> The $\max$ closure is single-count \emph{by construction}:
> $L_\mathrm{loss}^\mathrm{eff}/L_\mathrm{mech}=\max\!\big(L_\mathrm{cool}/L_\mathrm{mech},\,\theta\big)$
> never equals the forbidden sum $L_\mathrm{cool}/L_\mathrm{mech}+\theta$, because the two terms are
> estimators of the \emph{same} sink and $\max$ takes one, not both. A Monte-Carlo over
> $(L_\mathrm{cool}/L_\mathrm{mech},\theta)\in[0,1]^2$ ($5\times10^5$ draws) corroborates this: zero
> draws enter the $2\theta$ region (max counted-to-single ratio $=1.0$).

This both **softens** (MC demoted from proof to corroboration) and **strengthens** (adds the
by-construction argument the assertion was missing).

---

## Patch 4 — flag three literature numbers for a PDF check

Web-verification confirmed the *physics* the note leans on (El-Badry+19 catastrophic cooling;
Lancaster+21 near-complete wind-energy cooling; Pittard, Fielding, Tan/Oh/Gronke mixing layers). Three
specific **numbers**, however, should be checked against the actual PDFs before the note ships — mark
each with a `% TODO verify against PDF`:

1. **El-Badry+2019, Eq. 35** — the cooling-efficiency / evaporation expression the note cites for the
   density scaling. Confirm equation number and the exact prefactor/exponent.
2. **El-Badry+2019, Fig. 7** — the panel the note cites for the $3$–$30\times$ evaporation
   suppression. Confirm it is Fig. 7 (not 6/8) and that the quoted factor range matches.
3. **Lancaster+2025, Eq. 39** — the fractal-area / cooling-fraction relation. Confirm the equation
   number and that the year/citation key (`2021` vs `2025`) is the intended paper.

---

### Provenance
All numbers from committed CSVs; regenerate from the repo root:
```bash
python docs/dev/transition/pdv-trigger/data/make_fmix_table.py        # -> data/fmix_table.csv
python docs/dev/transition/pdv-trigger/data/make_doublecount_mc.py    # -> data/doublecount_mc.csv
```
The `f_mix` table is a frozen screen; the live confirmation is the matched-`t` runs in `runs/`
(PLAN.md §Task B), in progress.
