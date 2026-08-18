# TRINITY's force budget vs. the literature: $P_b$, $P_{\rm ram}$, $P_{\rm HII}$, $P_{\rm drive}$

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

**Status (2026-08-18):** 📘 reference — an external assessment of TRINITY's force budget against
five papers, contributed by the maintainer. Its §4 items are candidate work, not decisions; §4.1
(`alpha_p`) and §4.2 (the CEM coupled closure) are the two that bear directly on the open D5
question in `phii-identity/PLAN.md`.

> ⛔ **A CORRECTION IS INBOUND — do not act on §4 (2026-08-18).** The maintainer flagged that this
> revision "might contain wrong information" and is preparing an updated one. Until it lands, treat
> every §4 item as **unverified motivation, not a finding**. Nothing in the workstream's ledger
> depends on it — see the standing rule in `PLAN.md` §0 C-0.5 — and the four PLAN.md lines that cite
> it (1717, and Batch 12's rationale) are attribution and motivation only, so no measured result
> moves if §4 turns out wrong.
>
> **Provenance and standing.** Written 2026-08-18 by an external reviewer with read-only access to
> the tree, and handed to this workstream by the maintainer. **Retained in this folder under the
> C-0 external-document carve-out** that `PLAN.md` §0 adds on the same date (maintainer ruling): it
> is an input we did not author, kept verbatim, never load-bearing. **It is not this workstream's own
> measurement**, so the ⚠️ banner applies with full force: its verification appendix (§6) is the
> author's own, and §6 itself states that nothing requiring a TRINITY run was verified. Where its
> claims have since been checked or contradicted by Batch 11, that is recorded in the
> "B11 cross-check" section immediately below — **read that before acting on §4.**

## B11 cross-check — 2026-08-18, what this workstream verified, corroborated and corrected

Added by Batch 11 on the day the document arrived. Nothing else in the document was altered.

**Independently verified against source (✅):**
- §1's `P_drive` table for the momentum phase — `P_drive = P_HII + P_ram`
  (`run_momentum_phase.py:265`) — and its note that `params['Pb']` is `P_ram` there
  (`run_momentum_phase.py:585,669,891`). B11.0 checked call-time ordering in all four runners.
- §4.3's sharpest point: `shell_structure.py:243-244` really does compute
  `_vol_ion = R_IF**3 - rShell0**3`, i.e. **trinity already subtracts the wind cavity in
  `n_IF_Str` and does not in `get_phii_c3c`**. That is an internal inconsistency in the shipped
  code, not just a difference from the literature.
- §4.7.4's "the `n_IF_Str > 0` gate is now vestigial" — same finding as B11.E.

**Corroborated by the papers themselves (✅), and this matters for §6b seam C:**
- **Four independent sources say the recombination balance excludes the wind cavity, and C3a is the
  only outlier.** Lancaster Paper I `eq:ionreceq2`: `(4π/3)(R_i³ − R_w³) α_B n_Hi² = Q_0`, with the
  paper noting "the WBB enhances `ρ_i` relative to the classical solution, due to the presence of
  `R_w` in the denominator". Geen et al. 2019 `wind:photoequilibrium`: the same. Geen & de Koter 2022
  `eqn:photoionisation_equilibrium_uniform`: `Q_H = (4π/3) n_i² (r_i³ − r_w³) α_B`, again the same.
  And trinity's own `shell_structure.py:243`: `_vol_ion = R_IF**3 - rShell0**3`. `get_phii_c3c`
  balances over `(4/3)πR2³`, i.e. over the volume all four exclude.
- **§2.3's claim that the shell inner boundary condition matches Geen 2022 is exact.** Their
  `eqn:PwPibalance` is `P_w = P_i ≡ (m_H/X) n_i c_i²`, integrated "beginning at `r_w`, where the
  density `n_i` is set by the pressure balance with the wind bubble" — which is
  `shell_structure.py:125-126` verbatim. This independently supports **B11.0's revision of §6b seam
  B**: the shipped `nShell0 ∝ Pb` is the standard closure, so B11.B must measure the inconsistency
  rather than "correct" the shell.
- Lancaster Paper I `eq:pr_spitzer_adj` explicitly adopts `M_sh = 4π R_i³(ρ̄ − ρ_i)/3`, "that is, to
  subtract out the mass in ionized gas… Though this reduction is not consistent with the derivation
  of `eq:HIImomentum2`, we will see that it can be more accurate." So §4.4's proposal is one the
  literature already makes, with the same stated caveat.

**Corrected by measurement (⛔ — do not act on §4.4 as written):**
- §4.4 says subtracting the ionised mass is "**currently near-irrelevant** (your ionised layer is a
  thin skin)". **That is false in the momentum phase.** B11.0 measured `dR_ion/R2` = **0.658–1.308**
  with `dR_ion/dR_full` median **0.9954** — the momentum shell is essentially **entirely** ionised,
  not a skin. B11.C2 then measured the adjustment's cost: debiting `M_cav(t)` moves `R2(t=1.5)` by
  **+8.55%** (inertia only) or **+9.22%** (inertia and gravity), against an offline control that
  reproduces the run to 0.871%. So §4.4 is **first-order today**, not "the moment §4.3 lands", and
  it should be promoted in §7's ordering.
- §2.1(a)/§4.1's premise that the confined branch fires in 1a/1b "on every energy and implicit row
  of all five configs" holds **at nominal wind only**. `data/b7_confinement_screen.csv` records
  `B3MW001` (`Lw × 0.01`) at **78.4% HII-dominated in the energy phase** (ratio 0.487–4.927). The
  claim needs the qualifier.

**Open and worth testing (🔵):** §4.1's `alpha_p` hypothesis is the document's own highest-impact
item and §6 correctly labels it untested. Note what it cannot do: in a **weak-wind** regime
`alpha_p·P_ram` is small whatever `alpha_p` is, so the low-wind rungs isolate the C3a question from
the `alpha_p` question. See `PLAN.md` §Batch 12.

---

**Assessment date:** 2026-08-18
**Code state read:** `~/unsync/Code/Trinity` (read-only; nothing under `trinity/` was modified)
**Papers read in full:** Lancaster et al. Paper I (`lancaster2025.tex`), Lancaster et al. Paper II (`lancaster2025b.tex`), Geen et al. 2019 (`geen2019.tex`), Geen & de Koter 2022 (`geen2022.tex`), Haid et al. 2018 (`Haid118_arxiv.tex`)

Equation/figure references use the papers' own LaTeX labels (grep the `.tex`), not printed numbers.
Every quantitative claim was re-verified symbolically or numerically; §6 lists what and how, and what
I could *not* verify.

---

## 0. Verdict in three sentences

**You are doing this correctly, and in several respects better than any of the five papers.** The C3c
regime switch is not just defensible — in the momentum phase its crossover point is *exactly*
Lancaster's characteristic radius $R_{\rm ch}$ (verified symbolically), which means you independently
rederived the central scale of their Co-Evolution Model from a completely different starting point.
The two places where you can be measurably better are (i) the hard-coded $\alpha_p \equiv 1$ in the
ram-pressure term, which I believe is the actual resolution of your own open `momentum`-phase
question, and (ii) replacing the `max(...)` / `+` composition of $P_b$ and $P_{\rm HII}$ with
Lancaster's single coupled expression, which removes both the double-count and the phase-to-phase
sign flip in the systematic error.

---

## 1. What TRINITY actually computes (verified against source)

| Quantity | Source | Expression |
|---|---|---|
| $P_b$ (energy/implicit) | `get_bubbleParams.bubble_E2P` | $(\gamma-1)E_b\,/\,[\tfrac{4\pi}{3}(R_2^3-R_1^3)]$ |
| $R_1$ | `get_bubbleParams.get_r1`/`solve_R1` | root of $\sqrt{L_{\rm mech}(R_2^3-R_1^3)/(v_{\rm mech}E_b)}=R_1$ |
| $P_{\rm ram}$ | `get_bubbleParams.pRam` | $L_{\rm mech}/(2\pi R_2^2 v_{\rm mech})$, with $v_{\rm mech}=2L/\dot p$ |
| $P_{\rm HII}$ | `get_bubbleParams.get_phii_c3c` | $\dfrac{\mu_c}{\mu_i}k_BT_i\sqrt{\dfrac{3Q_{i,\rm abs}}{4\pi\chi_e\alpha_B R_2^3}}$ if $>P_{\rm conf}$, else **exactly 0** |
| $P_{\rm conf}$ | `get_bubbleParams.py:365` | `params['Pb']` — the *bubble thermal* pressure in 1a/1b/1c, the *ram* pressure in phase 2 (`run_momentum_phase.py:585,669`) |
| $P_{\rm ext}$ | `energy_phase_ODEs.get_press_ion` | $\frac{\mu_c}{\mu_i}n(r_{\rm sh})k_BT_i$ when $f_{\rm abs,ion}<1$, $+P_{\rm ISM}k_B$ beyond $r_{\rm cloud}$ |
| $P_{\rm drive}$ (1a/1b) | `energy_phase_ODEs.py:256,388`, `run_energy_implicit_phase.py:532` | $\max(P_b,\,P_{\rm HII})$ |
| $P_{\rm drive}$ (1c) | `energy_phase_ODEs.py:253,385`, `run_transition_phase.py:331` | $\max(P_b,\,P_{\rm HII}+P_{\rm ram})$ |
| $P_{\rm drive}$ (2) | `run_momentum_phase.py:265,445` | $P_{\rm HII}+P_{\rm ram}$ |
| Shell momentum | `energy_phase_ODEs.py:263` | $M_{\rm sh}\dot v_2 = 4\pi R_2^2(P_{\rm drive}-P_{\rm ext})-\dot M_{\rm sh}v_2-F_{\rm grav}+F_{\rm rad}$ |

Two identities that fall out of this and are worth having in your head (both verified symbolically —
see §6):

**(I) TRINITY's $R_1$ closure is exactly free-wind ram-pressure balance.** Substituting the `get_r1`
root into `bubble_E2P` gives

$$P_b \;=\; (\gamma-1)\,\frac{3}{4\pi}\,\frac{L_{\rm mech}}{v_{\rm mech}R_1^2}
\;\;\xrightarrow{\;\gamma=5/3\;}\;\;
\frac{L_{\rm mech}}{2\pi v_{\rm mech}R_1^2}\;=\;\frac{\dot p}{4\pi R_1^2}.$$

(Exact at the default `gamma_adia = 5/3`; the clean $\dot p/4\pi R_1^2$ form does depend on it, and
`gamma_adia` is user-exposed.) So the wind force reaching the shell in the energy phase is
$F = 4\pi R_2^2 P_b = \dot p\,(R_2/R_1)^2$: **TRINITY already carries a time-dependent momentum
enhancement factor**, it is just never named.

**(II) `pRam` sets $\alpha_p \equiv 1$ exactly.** With $v_{\rm mech}=2L/\dot p$
(`update_feedback.py:181`), $P_{\rm ram}=\dot p/(4\pi R_2^2)$, so $F_{\rm ram}=\dot p$ — the pure
Steigman (1975) limit, no enhancement.

---

## 2. Where you sit relative to each paper

### 2.1 Lancaster Paper I — the direct comparison

Paper I levels **three** criticisms at WARPFIELD. TRINITY's score is different on each.

**(a) "the force due to this thermal pressure gradient is not included in the momentum evolution
equation"** (§`subsec:theory_review`), verbatim:

> *"While the thermal pressure of the photoionized gas is used to solve for the hydro-static
> equilibrium shell, the force due to this thermal pressure gradient is not included in the momentum
> evolution equation \[see Rahner+17, Eqs 5-9\]."*

**Structurally fixed.** `include_PHII` + `get_phii_c3c` put a photoionised term into $P_{\rm drive}$ in
all four phases, and in the transition and momentum phases it is the dominant term in every config
you have measured. **Numerically it has not yet bitten in 1a/1b**: your own ledger (`PLAN.md`
Batch 5) records `frac HII-dominated = 0.0000` on every energy and implicit row of all five configs —
the confined branch fires everywhere there, `get_phii_c3c` returns exactly `0.0`, so
$P_{\rm drive}=P_b$ and the photoionised gradient contributes nothing to $\dot v_2$ in the
energy-driven phases. Whether that is physics or an artefact of the thin-shell geometry is §5's
question. So: *the machinery is there and demonstrably active downstream; the criticism is answered
structurally, and in 1c/2 numerically, but not yet in 1a/1b.*

**(b) "does not include a model for turbulently enhanced cooling ... which could make the WBBs act in
a more momentum-driven manner much earlier in the evolution even when conduction is weak"** — **still
open.** This is the gap your `cooling_boost_fA` / `cooling_boost_kappa` / `kappa-3way` workstream is
attacking. Paper I's "Prospects for Future Work" says the same thing about WARPFIELD, in a sentence
that is easy to misread if you stop at the comma:

> *"The ideal version of these semi-analytic models would include a parameterized model for this heat
> dissipation that is solved on-the-fly and included in the WBB energy equation... This is done for
> the case of only conductive heat dissipation in a spherical scenario by the* `WARPFIELD` *models,
> **but as \citet{Lancaster21a,Lancaster21c} has shown, cooling in turbulently-mixed
> intermediate-temperature gas can certainly dominate energy losses.**"*

So TRINITY has the *architecture* Paper I asks for (an on-the-fly energy equation with a resolved loss
integral) and half the *physics* (conduction). The turbulent-mixing half is exactly what your
$f_A$/$f_\kappa$ knobs are for. That framing is defensible and I would use it.

**(c) "simplifying assumptions such as the thin-shell approximation which prevent a faithful
representation of the gas density and ionization structure"** (§`subsec:problems`) — **partly
answered**: your shell solver *does* resolve the internal density and ionisation structure (§2.3).
What it does not do is separate $\mathcal{R}_w$ from $\mathcal{R}_i$. See §4.3.

**The C3c switch point is Lancaster's $R_{\rm ch}$ — exactly, in the momentum phase.** Setting
$P_{\rm C3a}(R_2)=P_{\rm conf}(R_2)$ with $P_{\rm conf}=P_{\rm ram}$:

$$\frac{\dot p}{4\pi R_2^2}=\bar\rho c_i^2\!\left(\frac{R_{\rm St}}{R_2}\right)^{3/2}
\;\;\Longleftrightarrow\;\;
R_2=\frac{R_{\rm eq}^4}{R_{\rm St}^3}\equiv R_{\rm ch},$$

with $R_{\rm eq}^2 \equiv \alpha_p\dot p/(4\pi\bar\rho c_i^2)$ (Paper I `eq:RE_MD_def`, `eq:eta_def`,
`eq:Rch_def`). SymPy returns crossover$/R_{\rm ch} = 1$ exactly. So in phase 2:

- your **confined branch** ($P_{\rm HII}=0$) is precisely Lancaster's wind-dominated $R_2<R_{\rm ch}$;
- your **unconfined branch** is precisely their PIR-dominated $R_2>R_{\rm ch}$.

**Two caveats, both important.** First, `get_phii_c3c` compares against `params['Pb']`, which is
$P_{\rm ram}$ *only in the momentum phase*. In 1a/1b/1c it is the bubble **thermal** pressure — larger
than $P_{\rm ram}$ by $(R_2/R_1)^2$ — so there the switch is the energy-driven analogue of
$R_{\rm ch}$: the same force-balance *structure* Lancaster use, but not the quantity they tabulate
(they define $R_{\rm ch}$ only for the MD case). That is a natural generalisation, not an error, but
it does mean `t_cross` in your ladder is **not** the $R_2=R_{\rm ch}$ crossing. Second, the reduction
assumes $f_{\rm abs,ion}=1$; since $P_{\rm C3a}\propto\sqrt{f_{\rm abs}}$, partial absorption shifts
the crossover by $f_{\rm abs}^{1/3}$.

With those caveats the correspondence stands and is worth stating in the method paper: you got there
from a confinement argument, Lancaster from force balance, and the momentum-phase scales agree
exactly. Batch 8's result that C3a reproduces Hosokawa–Inutsuka to 0.0000% is the same statement seen
from the Spitzer side.

**Where TRINITY is *better* than the CEM:** the CEM ignores gravity, external pressure, direct and
indirect radiation pressure, the cloud density profile, on-the-fly bubble cooling, and time-variable
feedback. TRINITY has all seven.

**Where the CEM is better:** it separates $\mathcal{R}_w$ from $\mathcal{R}_i$. TRINITY's $R_2$ does
both jobs. That is the root of every remaining discrepancy below.

### 2.2 Lancaster Paper II — the calibration you are missing

Paper II's CEM-comparison table (`tab:cem_comp`) measures $\langle\alpha_p\rangle = 4.57$–$6.82$ in
3D RMHD, and Paper I adopts $\alpha_p=6.25$ for its fiducial cluster. TRINITY's momentum phase uses
$\alpha_p=1$. Paper II also finds photoionised gas **increases** $\alpha_p$ relative to wind-only runs
(2.55 → 4.66 HD, 4.09 → 6.20 MHD), by removing collisional Ly$\alpha$ as an interface coolant *and* by
smoothing background inhomogeneity so the interface area $A_w$ drops. Note the sign: adding
photoionisation makes the *wind* more effective, not less. See §4.1.

### 2.3 Geen & de Koter 2022 — you already implement their model, numerically

`get_shellODE.py` is Geen 2022's `draine1`–`draine3` — the Draine (2011) hydrostatic
dusty-ionised-shell system with $\phi$ and $\tau$ integrated through the shell — with the same inner
boundary condition $P_w = P_i = (m_H/X)n_ic_i^2$ (`eqn:PwPibalance`; TRINITY
`shell_structure.py:125`). Their "overflow" criterion (the shell can no longer absorb $Q_H$) is your
`is_phiDepleted == False` / `f_esc_ion > 0` exit. You solve it self-consistently along the
trajectory; they solve it on a prescribed $r_w(t)$. **You are strictly ahead here.**

Two things they have that you don't: their $\Omega < 4\pi$ solid-angle geometry (your `coverFraction`
is an energy leak, not a solid angle — different physics), and an analytic overflow radius
(`eqn:overflowcondition`) that would make a good independent check on your numerical criterion. Note
their sign result: for $\omega<5/4$ overflow becomes *less* likely with radius (trapping wins), for
$\omega>5/4$ *more* likely. TRINITY's `densPL_alpha` spans both sides of that boundary, so it is a
sharp, cheap validation target.

### 2.4 Geen et al. 2019 — you are better on the one thing that matters

Geen 2019's dynamical closure is a *ram-pressure balance* at the front,
$n_ic_i^2 = n(r_i)(\dot r_i+v_0)^2$, with no shell inertia. Lancaster Paper I
§`subsec:theory_review` flags this as the model's key weakness (citing Raga+2012b). **TRINITY
integrates the full thin-shell momentum equation including $\dot M_{\rm sh}v_2$ and
$M_{\rm sh}\dot v_2$** — Hosokawa–Inutsuka rather than Spitzer. That is the right call and you should
say so.

Geen 2019 also gives you a cross-check quantity: $C_w \propto \dot p_w^{3/2}Q_H^{-3/4}c_i^{-3}$
(`wind:condition`), $C_w\approx 0.0093$ at fiducial values. Same wind/photoionisation ordering as
Lancaster's $\zeta$ but different exponents; computing both from TRINITY output would be a cheap,
high-value figure.

### 2.5 Haid et al. 2018 — the sanity anchor, and a genuine tension worth knowing about

Haid's headline is that **the ambient medium, not the star, decides the winner**: radiation dominates
by $\sim\!50\times$ in CNM ($n_0=100$), winds dominate by $10^2$–$10^4$ in WIM ($n_0=0.1$,
$T=10^4$ K), with the switch near $n_0\sim1\,$cm$^{-3}$.

Haid also measures the non-additivity directly, and **it goes the opposite way to Lancaster's
analytic result** — worth knowing before you cite either:

> *"In the CNM, the feedback from both processes $p_{\rm Combi}$ is larger than
> $p_{\rm IRad}+p_{\rm Wind}$ by $\sim$ 1, 3, and 23 percent... In the WIM, the difference is a factor
> of $\sim$ 3.2, 2.8, and 1.9."*

i.e. Haid find the naive sum is a **lower** limit, while Lancaster's CEM (`fig:force_comp`) finds the
sum **over**-estimates by $\sim$35% near $R_{\rm ch}$. Both are right and they are not measuring the
same thing: Lancaster's is a statement about *pressure geometry* (the WBB and PIR act at different
radii and compress each other, reducing the net force), Haid's is about *coupled microphysics* in a
simulation (each process makes the other couple better) — and Paper II
§`subsec:PIR_cooling_effect` independently confirms Haid's direction by measuring $\alpha_p$ rise when
LyC is switched on. **Net:** the sum is wrong for two reasons that partly cancel, which is precisely
why a coupled closure (§4.2) beats picking either `max` or `+`.

Haid use a temperature-dependent $\alpha_B = 2.56\times10^{-13}(T/10^4)^{-0.83}$ and let $T_i$ emerge
from the chemistry: 7160–8150 K across their grid (varying with *both* stellar mass and ambient
phase). TRINITY pins `caseB_alpha = 2.59e-13` and `TShell_ion = 1e4` as run constants
(`registry.py:396,400`). See §4.5.

---

## 3. What is unambiguously right

1. **Shell momentum equation.** $\mathrm{d}(Mv)/\mathrm{d}t$ form with the $\dot M v$ term. Correct,
   and better than Geen 2019/2020's pressure-balance closure.
2. **The "confined skin transmits" argument behind C3c.** Right — and Lancaster's CEM shows it is
   *more* right than you claimed. See §4.2.
3. **$R_1$ from free-wind ram balance.** Weaver's own condition. (Lancaster instead use the
   strong-shock post-shock value $\tfrac34\rho v^2$, so their
   $P_{\rm hot}=3\dot p/(16\pi\mathcal{R}_f^2)$ is $3/4$ of yours at the same $\mathcal{R}_f$. Both
   conventions are in the literature; just don't mix them when quoting $\alpha_p$ — §4.7.)
4. **Continuity of the energy→momentum handover.** Because $P_b\to\dot p/(4\pi R_1^2)$ and
   $R_1\to R_2$ as $E_b\to0$, $P_b\to P_{\rm ram}$ automatically, so `max(P_thermal, P_ram)` in the
   transition phase is a genuine continuous handover, not a patch. Your own seam measurements
   (`PLAN.md`, ratios 0.995–0.999) confirm it. Elegant; draw attention to it.
5. **Self-consistent radiation trapping.** You compute $f_{\rm esc,ion}$ from the actual shell
   structure instead of Lancaster's threshold-density formula (`eq:phot_trap_rhobar`) or Geen 2022's
   prescribed $r_w(t)$.
6. **$P_{\rm ext}$ and $P_{\rm ISM}$.** Lancaster Paper I explicitly drops external pressure
   (§`sec:theory_joint`: *"we ignore ... any external pressures or the effects of gravity"*). You keep
   both, with the right sign.

---

## 4. What can be better — ranked by expected impact

### 4.1 (highest) $\alpha_p \equiv 1$ is almost certainly why your momentum phase is universally HII-dominated

Your ledger (`PLAN.md`, Batch 5 s3) records the momentum drive as
$P_{\rm C3a}+P_{\rm ram} = 2.4$–$4.3\times$ stock's $2P_{\rm ram}$, i.e.

$$\frac{P_{\rm C3a}}{P_{\rm ram}} \;=\; 3.8 \;\text{–}\; 7.6,$$

and that inverting it would need $L_w\approx260$. But $\alpha_p$ multiplies $P_{\rm ram}$ *directly*,
without touching the SPS luminosity:

| $\alpha_p$ | $P_{\rm C3a}/(\alpha_p P_{\rm ram})$ |
|---|---|
| 1 (current) | 3.80 – 7.60 |
| 3 | 1.27 – 2.53 |
| 4.66 (Paper II `HWR`) | 0.82 – 1.63 |
| 6.20 (Paper II `MWR`) | **0.61 – 1.23** |

At Paper II's measured values, roughly half the measured range crosses below unity. Your comment
*"an inversion would need an unphysical $L_w\sim260$... it is NOT an O(1) normalisation error"* is
correct **and** the missing factor is not in C3a's normalisation at all — it is a missing factor in
$P_{\rm ram}$.

**Caveat on how this would actually play out.** Because `params['Pb']` *is* $P_{\rm ram}$ in the
momentum phase, applying $\alpha_p$ to the wind also raises $P_{\rm conf}$. So wherever
$\alpha_p P_{\rm ram} > P_{\rm C3a}$, `get_phii_c3c` returns exactly `0.0` and
$P_{\rm drive}=\alpha_p P_{\rm ram}$ — a hard **branch flip**, not a smooth blend of two comparable
terms. The result is a bimodal momentum phase (fully wind-driven or fully HII-driven, nothing in
between), which is itself an argument for §4.2.

Equivalently in Lancaster's parameterisation, with your Paper-II-matched numbers:

| $\alpha_p$ | $\zeta=R_{\rm eq}/R_{\rm St}$ |
|---|---|
| 1 | 0.40 |
| 3 | 0.69 |
| 6.2 | 0.99 |

**Two ways to set it, in increasing order of ambition.**

*(a) As an explicit parameter.* Add `alpha_p` (default 1.0 for byte-identity) and use
`pRam(...) * alpha_p` wherever the wind pushes the shell. One knob, pre-registered ablation, done.

*(b) Self-consistently from $\theta$ at the handover.* Better, and it costs nothing new. Lancaster
Paper I `eq:tofap` relates the two:

$$1-\theta_{{\rm MD},\alpha}(t)=\frac34\left(\frac{3}{2\pi}\frac{\alpha_p^5\dot p}{\bar\rho\,\mathcal{V}_w^4 t^2}\right)^{1/4}
\;\Longrightarrow\;
\alpha_p=\left[\frac{2\pi}{3}\frac{\bar\rho\,\mathcal{V}_w^4 t^2}{\dot p}\right]^{1/5}\left[\tfrac43(1-\theta)\right]^{4/5}.$$

TRINITY already measures $\theta = L_{\rm loss}/L_{\rm gain}$ — it *is* your transition trigger
(`phaseSwitch_LlossLgain = 0.05` $\Rightarrow \theta=0.95$). Evaluating the inverse at the Paper-II
cluster parameters ($\bar\rho=2.98\,M_\odot$pc$^{-3}$, $\mathcal{V}_w=3230$ km/s,
$\dot p=4.79\times10^4\,M_\odot$ km/s/Myr):

| $t$ [Myr] | $\theta=0.90$ | $\theta=0.95$ | $\theta=0.98$ |
|---|---|---|---|
| 0.1 | 8.6 | **5.0** | 2.4 |
| 0.3 | 13.4 | **7.7** | 3.7 |

So your own handover criterion, read through `eq:tofap`, implies $\alpha_p\approx5$–8 at handover —
bracketing Paper II's measured 4.57–6.82. That makes $\alpha_p$ a *derived* quantity rather than a
fitted one. (Caveat: `eq:tofap` assumes a momentum-driven scaling and $\alpha_p\gtrsim1.25$, and
$\alpha_p\propto t^{2/5}(1-\theta)^{4/5}$ — an estimate, not a precision result. Freeze it at handover
rather than evolving it.)

**One honest limitation.** Lancaster 2024a's microphysical formula (Paper I `eq:alphap_derive`),
$\alpha_p = \tfrac34\frac{\mathcal{V}_w/4}{\langle v_{\rm out}\rangle}\frac{4\pi\mathcal{R}_w^2}{A_w}$,
has $A_w \equiv 4\pi\mathcal{R}_w^2$ in 1D — so a 1D code *cannot* derive $\alpha_p$ from geometry;
the fractal-area factor is the whole point and it is unavailable to you. Route (b) sidesteps this by
going through the energy budget instead, which is legitimate but should be labelled a closure, not a
derivation.

### 4.2 Replace `max(...)` / `+` with Lancaster's single coupled expression

Highest value per line of code.

**Your open question**, from `get_phii_c3c`'s docstring and `PLAN.md`:
> *"$P_{\rm C3a}\propto R_2^{-3/2}$ vs $P_{\rm ram}\propto R_2^{-2}$: does a real momentum-phase
> cavity stay Strömgren-filled?"*

**Lancaster answers it, and the answer generalises your own C3c intuition.** In the co-evolution
phase the ionised gas is *always* in pressure balance with the wind bubble (Paper I `eq:force_cond`).
So the pressure at the shell is *always* the wind-bubble pressure
$\alpha_p\dot p/(4\pi\mathcal{R}_w^2)$ — photoionisation never adds an independent pressure. What it
changes is **where that pressure acts**: recombination balance in the cavity-corrected volume
(`eq:ionreceq2`) gives (`eq:RiRw_rel`, `eq:Rch_def`)

$$\mathcal{R}_i=\mathcal{R}_w\left(1+\frac{\mathcal{R}_w}{R_{\rm ch}}\right)^{1/3},
\qquad
R_{\rm ch}=\frac{\alpha_B}{12\pi(\mu_H m_H c_i^2)^2}\frac{\alpha_p^2\dot p^2}{Q_0},$$

so the force on the shell is

$$\boxed{\;F_b=4\pi\mathcal{R}_i^2 P_i=\alpha_p\dot p\left(\frac{\mathcal{R}_i}{\mathcal{R}_w}\right)^{\!2}
=\alpha_p\dot p\left(1+\frac{\mathcal{R}_w}{R_{\rm ch}}\right)^{\!2/3}.\;}$$

Your "a confined skin transmits the confining pressure rather than adding to it" *is* this statement —
and Lancaster show it stays true even when the ionised gas is not a skin. I verified numerically that
this expression:

- $\to \alpha_p\dot p$ (your **confined** branch) as $\mathcal{R}_w/R_{\rm ch}\to0$ — 1.001 at $10^{-3}$;
- $\to F_{b,\rm Sp}=4\pi\bar\rho c_i^2R_{\rm St}^2(\mathcal{R}_i/R_{\rm St})^{1/2}$ (your **unconfined**
  branch) as $\mathcal{R}_w/R_{\rm ch}\to\infty$ — ratio 1.0005 at $10^{3}$, 1.0000 at $10^4$;
- and interpolates smoothly through the crossover.

**So your two branches are the correct asymptotes of the CEM.** The error is confined to the
crossover. Evaluating at the shell radius (TRINITY's convention $R_2 \equiv \mathcal{R}_i$):

| $\mathcal{R}_i/R_{\rm ch}$ | $F_{\rm sum}/F_{\rm CEM}$ (your **momentum** phase) | $F_{\rm max}/F_{\rm CEM}$ (your **1a/1b** phases) |
|---|---|---|
| 0.5 | +34% | −22% |
| **1.0** | **+34%** | **−33%** |
| 2.0 | +32% | −23% |
| 10 | +21% | −8% |

(Lancaster quote the $\sim$35% over-estimate at $\mathcal{R}_i/R_{\rm ch}\approx1$, `fig:force_comp` —
my +34% is an independent reproduction, which is the check that my normalisation is right.)

The point is not a discontinuity: your seam ratios (0.995–0.999) show $P_{\rm drive}$ is continuous in
practice. The point is that **the model's systematic bias changes sign between phases** — $-33\%$
where `max` is used, $+34\%$ where `+` is used, worst exactly at the crossover. A single coupled
closure removes both at once and makes all four phases consistent for the first time.

**Implementation.** You already have the pattern — `solve_R1` brackets a scalar root on $[0,R_2]$:

```
# given R2 (= R_i, the shell), solve R2 = Rw*(1 + Rw/Rch)^(1/3) for Rw in (0, R2]
Rw = brentq(lambda x: x*(1.0 + x/Rch)**(1/3) - R2, 0.0, R2)
P_drive = alpha_p * pdot / (4*np.pi*Rw**2)
```

This replaces `max(Pb, P_HII)`, `max(Pb, P_HII + P_ram)` **and** `P_HII + P_ram`. In the energy phase
keep $P_b$ from the bubble solve in place of $\alpha_p\dot p/(4\pi\mathcal{R}_w^2)$ — the ED analogue
of the same closure (Paper I §`subsec:ed_jfb`).

Also worth knowing, in the weak-wind limit (`eq:approxF1`):
$F_b\approx F_{b,\rm Sp}+\tfrac{\alpha_p\dot p}{2}\frac{\mathcal{R}_w}{\mathcal{R}_i}$ — the wind
contributes *half* its momentum flux, further reduced by $\mathcal{R}_w/\mathcal{R}_i$. Your momentum
phase currently credits it the full $\dot p$ at the shell radius, which is where the rest of the
over-count lives.

### 4.3 (structural) Give the wind bubble its own radius

§4.2 is really the minimal way to introduce $\mathcal{R}_w \ne R_2$. The full version is worth a
follow-up paper: promote TRINITY to a three-radius model
$R_1 < \mathcal{R}_w \le R_2 \equiv \mathcal{R}_i$, with $\mathcal{R}_w$ **algebraic** (no new ODE
variable). In the confined regime $\mathcal{R}_w\to R_2$ and nothing changes — backward-compatible by
construction. This is also the direct answer to Paper I criticism (c).

It removes a real internal contradiction in the momentum phase: `get_phii_c3c` assumes the ionised gas
fills the sphere of radius $R_2$ (hence $R_2^3$ in the denominator), while `pRam` assumes the wind
pushes directly on the shell at $R_2$ (hence $\mathcal{R}_w=R_2$). Both cannot be true, and adding
their pressures asserts both simultaneously. Geen 2019's cavity-corrected balance
($\tfrac{4\pi}{3}n_i^2(r_i^3-r_w^3)\alpha_B=Q_H$, `wind:photoequilibrium`) and Lancaster
`eq:ionreceq2` both subtract the wind cavity for exactly this reason; your own shell solver already
does it correctly (`shell_structure.py:243`, `_vol_ion = R_IF**3 - rShell0**3`) — C3c is the one place
it doesn't.

### 4.4 Shell inertia should exclude the ionised interior

TRINITY puts the entire swept mass inside $R_2$ into the shell. Lancaster Paper I `eq:pr_spitzer_adj`
and Paper II `app:spitzer_momentum` show that subtracting the ionised-gas mass,
$M_{\rm sh}=\tfrac{4\pi}{3}\mathcal{R}_i^3(\bar\rho-\rho_i)$, matches simulations much better than the
unadjusted form (`fig:sptiz_momentum_comp`). Currently near-irrelevant (your ionised layer is a thin
skin), but first-order the moment §4.3 lands. Cheap to add at the same time.

### 4.5 $T_i$, $c_i$ and $\alpha_B$ are metallicity-blind

`TShell_ion = 1e4` and `caseB_alpha = 2.59e-13` are `run_const=True` and are not derived from
`ZCloud` (`registry.py:396,400` — no validator, no resolver; nothing in the tree derives either from
metallicity). But:

- Geen 2019 compute $c_i$ from Cloudy as a function of $T_*$, $Z$, $n_i$, $\mathcal{U}$
  (`fig:ionisedsoundspeed`), $\sim$10 km/s at solar $Z$ rising toward $\sim$15 km/s at $Z=0.002$, and
  their wind-importance coefficient scales as $C_w\propto c_i^{-3}$ (`wind:condition`), the breakout
  coefficient as $C_B\propto c_i^{-4}$ (`breakout:condition`);
- Lancaster's $\zeta\propto c_i^{-1}$ (MD), $c_i^{-3/2}$ (ED);
- Haid measure $\bar T_{\rm HII}$ = 7160–8150 K and use $\alpha_B\propto T^{-0.83}$.

A factor 1.5 in $c_i$ is a factor $\sim$3 in $C_w$ and $\sim$5 in $C_B$. Since `ZCloud` is already a
first-class TRINITY parameter that switches SPS tracks and cooling tables, letting it also set $T_i$
(a small table, or the Geen 2019 fit) closes a real inconsistency — and moves the answer in the
direction all three papers agree on: **low $Z$ makes photoionisation relatively stronger**. Probably
your cheapest genuinely-new physics.

### 4.6 The overflow limit degrades in an unphysical direction

$Q_{i,\rm abs}=Q_i f_{\rm abs,ion}$, so $P_{\rm C3a}\propto\sqrt{f_{\rm abs}}$ decays to zero as the
shell becomes transparent, and the `n_IF_Str > 0` gate fails outright at $f_{\rm abs}=0$. Meanwhile
$P_{\rm ext}$ switches on at the *first* escaping photon (`energy_phase_ODEs.py:235-238`, condition
`shell_fAbsorbedIon < 1.0`). Three different thresholds on the same physical transition, and the net
effect over the sequence is that photoionisation goes from accelerating to decelerating — precisely
where Geen 2022 say a champagne flow begins.

Geen 2022 are careful here, and worth quoting accurately: they say *"reaching the overflow radius does
not guarantee an immediate and strong champagne flow"*, describe a rarefaction wave moving back from
$r_i$ toward $r_w$ that eventually disperses the shell, and state *"the precise behaviour of the HII
region after it reaches the overflow radius is mostly beyond the scope of this paper."* So they do
**not** claim 1D models break there — they decline to model it.

My suggestion is not to model the champagne flow either, but to **make the regime change explicit**:
an event in `phase_events.py` on $f_{\rm esc,ion}$ crossing a threshold, logged as a distinct fate /
`SimulationEndCode`, so a run coasting through overflow is visible in the output instead of looking
like ordinary deceleration. Geen 2022's `eqn:overflowcondition` gives you an analytic overflow radius
for $\rho\propto r^{-\omega}$ as an independent cross-check.

Related: $P_{\rm ext}$ is evaluated from the **unshocked** profile at $r_{\rm shell}$
(`get_press_ion(rShell, ...)`) but applied over $4\pi R_2^2$. Fine for a thin shell; check it when the
shell is thick.

### 4.7 Diagnostics (cheap, and they would strengthen a paper)

1. **Output $\alpha_p$.** You have $R_1$ and $R_2$; emit Lancaster Paper I `eq:alphap_shock`,
   $$\alpha_p=\tfrac14\!\left[\,3\left(\tfrac{R_2}{R_1}\right)^{2}+\left(\tfrac{R_2}{R_1}\right)^{-2}\right],$$
   per snapshot — a direct, quantitative comparison against Paper II `tab:cem_comp`, for free. The
   single most persuasive validation figure available to you. **Do not** report $(R_2/R_1)^2$
   instead: it converges to $\tfrac43\times$ Lancaster's definition (1.251 at $R_2/R_1=1.5$, 1.3333 by
   $R_2/R_1=10$), because Weaver's $R_1$ condition uses the full ram pressure and Lancaster's the
   strong-shock post-shock value $\tfrac34\rho v^2$.
2. **Output $\zeta=R_{\rm eq}/R_{\rm St}$ and $R_2/R_{\rm ch}$.** Both one line from quantities you
   already carry; they place every TRINITY run directly on Paper I `fig:feedback_ratio` / Paper II
   `fig:feedback_ratio_comp`. Given that C3c's momentum-phase switch *is* $R_2=R_{\rm ch}$, `R2/Rch`
   is arguably the most informative single diagnostic your code could emit.
3. **`F_ram` is mislabelled in two phases.** `energy_phase_ODEs.py:415` and
   `run_transition_phase.py:338` both set `F_ram = Pb * 4πR2²` while `P_ram` is reported separately
   (and is 0 in the energy phase). Your `phii-identity/README.md` §8 flags the ramped-pressure version
   of this. Reporting only — but it is what every force-budget figure consumes.
4. **The `n_IF_Str > 0` gate is now vestigial.** Since C3c recomputes the density from $R_2$,
   `n_IF_Str` and its `min(n_IF_Str, shell_n0)` cap no longer feed $P_{\rm HII}$ — they only gate it.
   Either retire the gate or replace it with the condition it actually intends ($Q_{i,\rm abs}>0$,
   $R_2>0$), which `get_phii_c3c` already checks internally.

---

## 5. On the thin-shell / 1D approximation specifically

You flagged this as a constraint. Three observations:

**It is not the limitation you might think for $P_{\rm HII}$.** Everything in §4.2–4.3 is achievable
in 1D — Lancaster's entire CEM *is* a 1D thin-shell model. Separating $\mathcal{R}_w$ from
$\mathcal{R}_i$ costs one algebraic root-find, not a dimension.

**It *is* the limitation for $\alpha_p$.** In 1D $A_w\equiv4\pi\mathcal{R}_w^2$, so the fractal-area
enhancement Lancaster 2024a identify as the controlling physics is structurally unavailable. Any
$\alpha_p>1$ in TRINITY is a calibration and should be labelled as one. Your `cooling_boost_fA` /
`cooling_boost_kappa` knobs are the right hooks on the *energy* side; §4.1(b) is the corresponding
momentum-side closure, and the two should be made consistent — they are both statements about
$\theta$.

**The thin-shell geometry biases you toward "confined".** Putting *all* the swept mass in a thin shell
maximises the shell density and hence radiation trapping, relative to a turbulent, clumpy medium where
photons leak through low-density channels. Paper II quantifies exactly this: higher clumping
$\Rightarrow$ larger $\mathcal{R}_i$ and larger $p_r$, consistently across resolution and HD-vs-MHD
(§`subsec:role_of_turbulence`, `app:supp_analysis`). So your finding that *"the confined branch is the
one that fires in the energy and implicit phases"* may be partly geometric rather than physical —
which matters, because it is also the reason Paper I criticism (a) is not yet numerically answered in
1a/1b (§2.1). A clumping factor $\mathfrak{C}$ multiplying $\alpha_B n^2$ would let you test it for
almost no cost; Lancaster measure $\mathfrak{C}=2.39$–10.3 in their backgrounds (`subapp:clumping`).

---

## 6. Verification appendix

Everything quantitative above was checked, not asserted. SymPy/NumPy, all reproduced:

| Claim | Result |
|---|---|
| `solve_R1` root $\Rightarrow P_b=(\gamma-1)\tfrac{3}{4\pi}L/(vR_1^2)$; $=\dot p/(4\pi R_1^2)$ at $\gamma=5/3$ | exact identity (SymPy `simplify == 0`) |
| `pRam` with $v_{\rm mech}=2L/\dot p$ $\Rightarrow F_{\rm ram}=\dot p$ | exact |
| C3c crossover (momentum phase) $=R_{\rm eq}^4/R_{\rm St}^3=R_{\rm ch}$ | ratio $=1$ exactly |
| $F_{\rm CEM}=\alpha_p\dot p(1+\mathcal{R}_w/R_{\rm ch})^{2/3}\to\alpha_p\dot p$ | 1.001 at $\mathcal{R}_w/R_{\rm ch}=10^{-3}$ |
| $F_{\rm CEM}\to F_{b,\rm Sp}$ | ratio 1.0005 at $10^{3}$, 1.0000 at $10^4$ |
| $F_{\rm sum}/F_{\rm CEM}$ at $\mathcal{R}_i=R_{\rm ch}$ | 1.342 (Lancaster state $\sim$1.35) |
| $F_{\rm max}/F_{\rm CEM}$ at $\mathcal{R}_i=R_{\rm ch}$ | 0.671 |
| $(R_2/R_1)^2$ vs `eq:alphap_shock` | $\to4/3$ (1.3333 at $R_2/R_1=10$) |
| `eq:tofap` inverted for $\alpha_p$ | round-trips $\theta$ to $10^{-6}$ |
| $\zeta(\alpha_p=6.25)$, Paper II cluster | 0.991 (paper quotes $\approx0.98$) |
| $P_{\rm C3a}/P_{\rm ram}=3.8$–7.6 from `PLAN.md`'s "2.4–4.3× stock's $2P_{\rm ram}$" | $2\times2.4-1$, $2\times4.3-1$ ✓ |

Every file:line citation in §1 was re-checked against the staged source, and every LaTeX label against
the cited paper. An independent adversarial pass over this document caught sixteen issues, all
corrected above — most consequentially: the $R_{\rm ch}$ identification holds only in the momentum
phase; the Paper I "Prospects" quote continues past where I first cut it, and the continuation
reverses its sense; and Haid's non-additivity has the *opposite* sign to Lancaster's.

**What I did *not* verify:** anything requiring a TRINITY run. The $P_{\rm C3a}/P_{\rm ram}=3.8$–7.6
figure is inferred from your `PLAN.md` Batch 5 ledger, not measured by me, so §4.1's $\alpha_p$ table
is a hypothesis to test, not a result. The cheapest test is an offline screen on existing
momentum-phase trajectories — the trick your own Batch 5 stage 2 showed to be a trustworthy filter.
Geen 2019's $c_i(Z)$ range is supported by their figure and a commented-out table rather than
quotable text.

**Staleness caveat:** `docs/dev/` carries its own "may be out of date" banner. I took its *measured
numbers* at face value; every code claim above was re-checked against current source.

---

## 7. Suggested order of work

1. Emit $\alpha_p$ (`eq:alphap_shock` form), $\zeta$ and $R_2/R_{\rm ch}$ as diagnostics. Zero risk,
   immediately answers "where do my runs sit on `fig:feedback_ratio`".
2. Offline screen: recompute momentum-phase $P_{\rm drive}$ with $\alpha_p\in\{1,3,5,6.2\}$ on
   existing output. Confirms or kills §4.1 in an afternoon, no solver run.
3. If confirmed: add `alpha_p`, default 1.0; then the $\theta$-derived closure.
4. Replace the four $P_{\rm drive}$ expressions with the single CEM form (§4.2). Pre-register the
   ablation as you did for C3c — it will move fates.
5. $Z$-dependent $T_i$ (§4.5).
6. Paper-scale: the three-radius model (§4.3) + adjusted shell mass (§4.4) + a clumping factor (§5).

---

## Sources

- [`lancaster2025.tex`](computer:///Users/jwt/unsync/Code/Trinity) — Lancaster, J.-G. Kim, Bryan, Menon, Ostriker & C.-G. Kim, *The Co-Evolution of Stellar Wind-blown Bubbles and Photoionized Gas I* (project doc)
- [`lancaster2025b.tex`](computer:///Users/jwt/unsync/Code/Trinity) — *…II: 3D RMHD Simulations and Tests of Semi-Analytic Models* (project doc)
- [`geen2019.tex`](computer:///Users/jwt/unsync/Code/Trinity) — Geen, Pellegrini, Bieri & Klessen, *When HII Regions are Complicated* (project doc)
- [`geen2022.tex`](computer:///Users/jwt/unsync/Code/Trinity) — Geen & de Koter, *Bottling the Champagne* (project doc)
- [`Haid118_arxiv.tex`](computer:///Users/jwt/unsync/Code/Trinity) — Haid, Walch, Seifried, Wünsch, Dinnbier & Naab, *The relative impact of photoionizing radiation and stellar winds on different environments* (project doc)
- [`trinity/bubble_structure/get_bubbleParams.py`](computer:///Users/jwt/unsync/Code/Trinity/trinity/bubble_structure/get_bubbleParams.py)
- [`trinity/phase1_energy/energy_phase_ODEs.py`](computer:///Users/jwt/unsync/Code/Trinity/trinity/phase1_energy/energy_phase_ODEs.py)
- [`trinity/phase2_momentum/run_momentum_phase.py`](computer:///Users/jwt/unsync/Code/Trinity/trinity/phase2_momentum/run_momentum_phase.py)
- [`trinity/phase1c_transition/run_transition_phase.py`](computer:///Users/jwt/unsync/Code/Trinity/trinity/phase1c_transition/run_transition_phase.py)
- [`trinity/shell_structure/shell_structure.py`](computer:///Users/jwt/unsync/Code/Trinity/trinity/shell_structure/shell_structure.py)
- [`trinity/shell_structure/get_shellODE.py`](computer:///Users/jwt/unsync/Code/Trinity/trinity/shell_structure/get_shellODE.py)
- [`trinity/sps/update_feedback.py`](computer:///Users/jwt/unsync/Code/Trinity/trinity/sps/update_feedback.py)
- [`trinity/_input/registry.py`](computer:///Users/jwt/unsync/Code/Trinity/trinity/_input/registry.py)
- [`docs/dev/phii-identity/README.md`](computer:///Users/jwt/unsync/Code/Trinity/docs/dev/phii-identity/README.md), [`PLAN.md`](computer:///Users/jwt/unsync/Code/Trinity/docs/dev/phii-identity/PLAN.md)
