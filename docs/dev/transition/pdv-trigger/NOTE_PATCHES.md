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

Ready-to-paste edits for the maintainer's external `.tex` note (the `.tex` is not in this repo, so these
are drop-in snippets + a precise change list). Every number traces to a committed CSV under `data/` /
`runs/data/`; reproduce with the builder named in each patch.

> **STATUS 2026-06-25 — the workstream has CONCLUDED, and the conclusion REDIRECTS the note.** The note as
> drafted builds toward `θ_target(Da)` as the recommended closure. The live matched-`t` runs (now done, 4/4)
> plus a gate-validated real-Da replay **refute** that closure: a cooling-magnitude knob — constant *or*
> `Da`-coupled — is **not** the energy→momentum trigger for normal clouds; **blowout is**, and the boost's
> job is to fix cooling *magnitude*. The faithful `κ_eff` interface re-derivation is the principled endgame.
> **Patches 5–7 carry this redirect; Patches 1–4 stay valid** (the `f_mix` screen, the double-count algebra,
> the literature) but must be read through Patch 6. Sources: `FINDINGS.md`, `KAPPA_EFF_SCOPING.md`,
> `runs/data/live_compare.csv`, `data/da_{screen,replay}.csv`.

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
> cannot place the transition at blowout everywhere. (The density-coupled
> $\theta_\mathrm{target}(\mathrm{Da})$ closure this motivated has since been \emph{tested and refuted} —
> see Patch 6; the spread instead points to the $\kappa_\mathrm{eff}$ re-derivation.)

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

## Patch 4 — literature anchors: El-Badry verified; use the Lancaster plateau at GMC density

**Update.** The El-Badry+2019 specifics are now PDF-verified (in the maintainer's revised note): the
cooling-efficiency relation is **Eq. 37/38** — $\theta=(\mathcal{L}_\mathrm{int}/\dot E)/(11/5+\mathcal{L}_\mathrm{int}/\dot E)$
— **not** the earlier "Eq. 35" (which is the interface-profile relation); $\theta$ is **time-independent**
(fiducial sims at $t\approx3.5$ Myr); anchor $\theta\approx0.61$ at $\lambda\delta v=1$; evaporation
suppressed $3$–$30\times$ (Fig. 7). Update the citations accordingly and drop the old "Eq. 35" TODO.

**The anchor that matters for TRINITY.** TRINITY's clouds sit at GMC densities ($n\sim10^2$–$10^6$), where
the right comparator is **Lancaster's near-complete-cooling plateau, $\theta\approx0.90$–$0.99$ (roughly
flat)** — *not* El-Badry's $\sqrt{\rho}$ relation extrapolated downward. El-Badry+2019 is a
**supernova-superbubble** study at $n\sim0.1$–$10$; pushing its $\theta(n)$ to GMC cores is extrapolation
past calibration. So: drop any schematic $\theta_\mathrm{lit}(n)$ band built that way; if a band is drawn,
use the flat Lancaster value with a $\theta_\mathrm{max}<1$ ceiling and label it qualitative until
digitized. (Caveat: our session could not open the PDFs — the El-Badry specifics rest on the maintainer's
PDF check; verify `Lancaster 2021` vs `2025` Eq. 39 key.)

---

## Patch 5 — the frozen screen is now backed by LIVE matched-$t$ runs (4/4 configs)

The `f_mix` table (Patches 1–2) was a frozen-trajectory screen; it is now confirmed live. Four matched-$t$
edge configs (boosted vs `none`, separate processes; `runs/data/live_compare.csv`):

```latex
\begin{table}
\centering
\caption{Live matched-$t$ edge runs (boosted vs unboosted, separate processes). The boost's effect is
density-dependent: dense clouds fire cooling early, diffuse clouds blow out first.}
\begin{tabular}{lll}
\hline
Config ($n_\mathrm{core}$) & boost & live outcome \\
\hline
f1edge\_hidens ($10^6$)     & $\times2$         & fires cooling \emph{at birth} ($t{=}0.003$ Myr), before blowout \\
simple\_cluster ($10^5$)    & $\times2$         & blows out ($0.109$) \emph{then} transitions ($0.131$); $\Delta E_b$ up to $47\%$ \\
f1edge\_lowdens ($10^2$)    & $\times2,\times3$ & does \emph{not} fire by blowout ($\sim0.62$ Myr); $\Delta E_b$ $13$–$24\%$ \\
fail\_repro ($5\times10^9$) & $\times2$         & unaffected (heavy, PdV-dominated) \\
\hline
\end{tabular}
\end{table}
```

> Confirmed live: \textbf{no constant $f_\mathrm{mix}$ fires cooling across the density grid} — dense clouds
> transition by cooling, diffuse clouds by blowout. (The diffuse runs hit a wall-clock ceiling, so the
> statement is "not fired \emph{by} blowout", not "never fires".)

---

## Patch 6 — $\theta_\mathrm{target}(\mathrm{Da})$ was tested and REFUTED; the conclusion is the pivot

**This is the redirect — apply it wherever the note presents $\theta_\mathrm{target}(\mathrm{Da})$ as the
recommended endpoint.** The coupled
$\theta_\mathrm{target}(\mathrm{Da})=\theta_\mathrm{max}\,\mathrm{Da}/(1+\mathrm{Da})$ closure has been put
on trial and **does not work**:

- **A constant target is degenerate with the $0.95$ trigger.**
  $f_\mathrm{mix}=0.95/(L_\mathrm{cool}/L_\mathrm{mech})$ (Patch 1) is \emph{by construction} the boost that
  lifts the resolved loss to the $0.95$ threshold, so calibrating to a flat literature $\theta\approx0.95$
  is \emph{bit-identical} to it. A constant target — spelled $f_\mathrm{mix}$ or $\theta$ — adds nothing
  over the trigger TRINITY already has.
- **$\theta_\mathrm{target}(\mathrm{Da})$ refuted (gate-validated).** An offline Da-screen and a real-Da
  replay that re-runs TRINITY's own interface cooling per trajectory row (gate: reproduces the logged
  $L_\mathrm{cool}$ to $\le3.9\times10^{-5}$, interface zone bit-identical) both return \textbf{NO-GO}: the
  interface temperature is $\sim$constant across clouds ($\sim$21–23 kK, the $\Lambda$ peak), so the real
  $\mathrm{Da}$ collapses to $\propto(R_2/v_2)P_b$, is \emph{non-monotonic} in $n_\mathrm{core}$, and is
  $\mathrm{Da}\gg1$ everywhere at blowout — so $\theta_\mathrm{max}\mathrm{Da}/(1+\mathrm{Da})$ saturates to
  a constant. It neither orders the clouds by density nor fires them at blowout under one knob
  (`data/da_screen.csv`, `data/da_replay.csv`).

**Replacement conclusion** (paste over the note's "$\theta_\mathrm{target}(\mathrm{Da})$ is the endpoint"):

> For normal clouds the operative energy$\to$momentum handoff is \textbf{geometric blowout}
> ($R_2=R_\mathrm{cl}$), not cooling balance: at blowout the resolved loss ratio is only $0.25$–$0.70$, well
> short of $0.95$, and no scalar cooling target — constant or $\mathrm{Da}$-coupled — places the transition
> there across the density grid. The cooling boost's role is therefore to correct the cooling
> \emph{magnitude} (so $E_b$, $P_b$, and the evaporation are right \emph{through} the handoff), not to
> trigger it. The faithful coupling that makes the cooling fraction \emph{emerge} per cloud — and reproduces
> El-Badry's evaporation suppression a scalar cannot — is the $\kappa_\mathrm{eff}$ mixing-layer interface
> (Patch 7).

---

## Patch 7 — $\kappa_\mathrm{eff}$ is the principled endgame, and it is feasible (not just aspirational)

The note lists $\kappa_\mathrm{eff}=\max(\kappa_\mathrm{Spitzer},\kappa_\mathrm{mix})$ as an out-of-scope
ideal. A feasibility map (`KAPPA_EFF_SCOPING.md`, verified against source) shows it is bounded and
implementable — the note can say so:

> A faithful $\kappa_\mathrm{eff}$ is a re-derivation of $\sim$three functions of the bubble-structure
> solver (the conduction term in the temperature ODE, the closed-form near-front initial conditions, and the
> evaporation seed) plus a $\kappa_\mathrm{mix}$ model — \emph{not} a coefficient swap, but \emph{not} a
> rebuild of the self-similar $(\beta,\delta)$ machinery either (that solver carries no explicit
> conduction-law dependence and survives unchanged). The one hard constraint — already seen to break a naive
> post-hoc sink — is keeping the evaporative mass flux $\dot M>0$ self-consistent while the interface
> cooling rises; the change must live \emph{inside} the structure solve, where it can decouple enhanced
> cooling from suppressed evaporation.

**Update 2026-06-26 — Rung A (the structural probe) is built and the crux is measured.** A gated
`cooling_boost_kappa` ($f_\kappa$, default 1.0, byte-identical off) that inflates the Spitzer prefactor
$C\to f_\kappa C$ at all three sites confirms, on the stiff `f1edge_hidens` edge at matched $t$, exactly the
predicted obstacle: $f_\kappa{=}2$ raises $L_\mathrm{cool}$ ($\times1.23$–$1.38$) **and** $\dot M$
($\times1.08$–$1.17$) together — cooling and evaporation rise *with the same sign*, and a $2\times\kappa$
moves the loss ratio only $+0.05$–$0.10$ toward the 0.95 trigger. So the flat-prefactor knob cannot reach
the transition without runaway evaporation; only the state-coupled $\kappa_\mathrm{eff}$ (Rung B) can. The
probe thus **confirms Rung B is required, not optional**. Artifacts + table: `KAPPA_EFF_SCOPING.md` §6a,
`data/kappa_backreaction.csv`, `kappa_backreaction.png`.

**Rung B is now scoped on paper (`RUNGB_SCOPING.md`, 2026-06-26).** Two independent (adversarial)
verifications establish the design: the conductive flux at the front is *one quantity read twice*, so a
faithful $\kappa_\mathrm{eff}$ must **sever $\dot M$ from the front balance** (entrainment-set, $>0$ by
construction) while a mixing-layer $\kappa_\mathrm{eff}$ raises cooling localized to the $\sim10^5$ K band; the
mix-branch near-front IC is **numerical** ($\kappa_\mathrm{mix}\propto1/T$ is not front-regular) and
$\kappa_\mathrm{mix}$'s magnitude is the real model (an entrainment efficiency $\alpha_\mathrm{mix}\ll1$, since
literal $D_\mathrm{turb}=R_2v_2$ gives an absurd $T_\mathrm{cross}\sim10^{12}$ K). No production code touched.

---

### Provenance
All numbers from committed CSVs / live runs; regenerate from the repo root:
```bash
python docs/dev/transition/pdv-trigger/data/make_fmix_table.py        # -> data/fmix_table.csv
python docs/dev/transition/pdv-trigger/data/make_doublecount_mc.py    # -> data/doublecount_mc.csv
python docs/dev/transition/pdv-trigger/data/make_da_screen.py         # -> data/da_screen.csv
python docs/dev/transition/pdv-trigger/data/make_da_replay.py         # -> data/da_replay.csv  (slow; real solver)
python docs/dev/transition/pdv-trigger/data/make_kappa_backreaction.py  # -> data/kappa_backreaction.csv + ../kappa_backreaction.png
```
The `f_mix` table (frozen screen) is now confirmed by the matched-`t` live runs in
`runs/data/live_compare.csv` (4/4 configs). Full conclusion + the κ_eff scope: `FINDINGS.md`,
`KAPPA_EFF_SCOPING.md`, `PLAN.md` §Outcome & pivot.
