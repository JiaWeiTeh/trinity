# TRINITY's force budget vs. the literature: $P_b$, $P_{\rm ram}$, $P_{\rm HII}$, $P_{\rm drive}$

**Revision 2 — 2026-08-18.** This supersedes revision 1 of the same date. Revision 1 contained
errors of substance, not only of detail; §6.2 lists every one of them and what replaced it. The
largest are: **§4.1's $\alpha_p$ diagnosis was wrong** (TRINITY already carries $\alpha_p$, and
$\alpha_p\equiv1$ in the momentum phase is the *correct* 1D value, not a defect); **§2.5's
"Haid–Lancaster tension" does not exist**; **§2.3 inverted the Geen 2022 / WARPFIELD chronology**;
and **§4.4 was contradicted by your own measurement** (the momentum shell is ~99.5% ionised, not a
skin, and the mass adjustment is worth $+8.6$–$9.2\%$ in $R_2$).

**Code state read:** `~/unsync/Code/Trinity` (read-only; nothing under `trinity/` was modified).
**Papers read in full:** Lancaster et al. Paper I (`lancaster2025.tex`), Paper II
(`lancaster2025b.tex`), Geen, Pellegrini, Bieri & Klessen 2019 (`geen2019.tex`, *When H\,II Regions
are Complicated*), Geen & de Koter 2022 (`geen2022.tex`, *Bottling the Champagne*), Haid et al. 2018
(`Haid118_arxiv.tex`).
**Workstream docs read:** `docs/dev/phii-identity/{README.md, PLAN.md, LITERATURE_ASSESSMENT.md}` at
the 2026-08-18 pull, including Batches 11 and 12 and the B11 cross-check.

Equation references use the papers' own LaTeX labels, not printed numbers. §6.1 lists what was
verified and how; §6.3 what the second verification pass changed; §6.4 what could **not** be
verified.

---

## 0. Verdict

**The force budget is structurally sound and, on several specific points, ahead of all five papers.**
The C3c regime switch is defensible, and in the momentum phase its crossover point is *exactly*
Lancaster's characteristic radius $R_{\rm ch}$ **evaluated at $\alpha_p=1$** — which is the value
TRINITY's momentum phase carries, so the identification is self-consistent. You reached from a
confinement argument the scale they reach from force balance.

**The two highest-value improvements are the two your own Batch 11/12 have already measured**, not
the ones revision 1 nominated:

1. **The recombination balance volume.** `get_phii_c3c` balances $Q_{i,\rm abs}$ over
   $\tfrac{4\pi}{3}R_2^3$ — the wind cavity. Lancaster `eq:ionreceq2`, Geen 2019
   `wind:photoequilibrium`, Geen 2022 `eqn:photoionisation_equilibrium_uniform` **and TRINITY's own
   `shell_structure.py:243`** all balance over $r_i^3-r_w^3$, i.e. over the volume C3a uses,
   *excluded*. This is an inconsistency internal to TRINITY, not merely a difference from the
   literature, and it is what your seam-C mass double-book measures.
2. **Shell inertia.** Subtracting the ionised mass is worth $+8.55\%$ / $+9.22\%$ in $R_2(t=1.5)$ at
   nominal wind (B11.C2, against a control good to $0.871\%$). Lancaster Paper I already adopts this
   form (`eq:pr_spitzer_adj`), with its own consistency caveat.

**$\alpha_p$ is not the answer to the momentum-phase question, and revision 1 was wrong to say it
was.** §4.4 replaces that recommendation with the correct reading of what TRINITY already computes.

---

## 1. What TRINITY actually computes (verified against source)

| Quantity | Source | Expression |
|---|---|---|
| $P_b$ (energy/implicit) | `get_bubbleParams.bubble_E2P` | $(\gamma-1)E_b\,/\,[\tfrac{4\pi}{3}(R_2^3-R_1^3)]$ |
| $R_1$ | `get_bubbleParams.get_r1`/`solve_R1` | root of $\sqrt{L_{\rm mech}(R_2^3-R_1^3)/(v_{\rm mech}E_b)}=R_1$ |
| $P_b$ **as the ODE sees it** | `get_effective_bubble_pressure`, `:495–503` | same, but with $R_1\!\to\!\frac{t-t_{\rm SF}}{10^{-3}}R_1$ for $t\le t_{\rm SF}+10^{-3}\,$Myr |
| $P_{\rm ram}$ | `get_bubbleParams.pRam` | $L_{\rm mech}/(2\pi R_2^2 v_{\rm mech})$, with $v_{\rm mech}=2L_{\rm mech}/\dot p_{\rm tot}$ |
| $P_{\rm HII}$ | `get_bubbleParams.get_phii_c3c` | $\dfrac{\mu_c}{\mu_i}k_BT_i\sqrt{\dfrac{3Q_{i,\rm abs}}{4\pi\chi_e\alpha_B R_2^3}}$ if $>P_{\rm conf}$, else **exactly 0** (`:365`) |
| $P_{\rm conf}$ | `get_bubbleParams.py:365` | `params['Pb']` — bubble *thermal* pressure in 1a/1b/1c, the *ram* pressure in phase 2 (`run_momentum_phase.py:585,669,891`) |
| $P_{\rm ext}$ | `energy_phase_ODEs.get_press_ion` | $\frac{\mu_c}{\mu_i}n(r_{\rm sh})k_BT_i$ when $f_{\rm abs,ion}<1$, $+\,n_{\rm ISM}k_BT$ beyond $r_{\rm cloud}$ |
| $P_{\rm drive}$ (1a) | `energy_phase_ODEs.py:388` | $\max(P_b,\,P_{\rm HII})$ |
| $P_{\rm drive}$ (1b) | `run_energy_implicit_phase.py:532` | $\max(P_b,\,P_{\rm HII})$ |
| $P_{\rm drive}$ (1c) | `run_transition_phase.py:331` | $\max(P_b,\,P_{\rm HII}+P_{\rm ram})$ |
| $P_{\rm drive}$ (2) | `run_momentum_phase.py:265,445` | $P_{\rm HII}+P_{\rm ram}$ |
| Shell momentum | `energy_phase_ODEs.py:263` | $M_{\rm sh}\dot v_2 = 4\pi R_2^2(P_{\rm drive}-P_{\rm ext})-\dot M_{\rm sh}v_2-F_{\rm grav}+F_{\rm rad}$ |

**Which of these the solver actually integrates.** This matters for any patch scoped by the table.
The transition phase's RHS is `get_ODE_transition_pure` (`run_transition_phase.py:630`), which
delegates the momentum equation to `get_ODE_Edot_pure` (`:231-233`); so the **live** 1a/1b/1c drive
expressions are `energy_phase_ODEs.py:253` (the `current_phase == 'transition'` branch) and `:256`
(the `else` branch), and the live momentum one is `run_momentum_phase.py:445`. Three live sites.
`run_transition_phase.py:331`, `run_energy_implicit_phase.py:532`, `energy_phase_ODEs.py:388` and
`run_momentum_phase.py:265` all sit in `compute_forces_pure` / `compute_derived_quantities` and are
**reporting only** — they set `params['P_drive']` for the snapshot and never reach the integrator.
`energy_phase_ODEs.py:385` is genuinely unreachable: `compute_derived_quantities` is called from one
place (`run_energy_phase.py:242`) with `current_phase == 'energy'`, so its `'transition'` branch never
fires. The formulas are identical either way, so Table 1 is right as physics; it is the *edit surface*
that differs.

Two identities follow, both verified symbolically (§6.1):

**(I) TRINITY's $R_1$ closure is exactly free-wind ram-pressure balance, so the code already carries
a momentum enhancement factor.** Substituting the `get_r1` root into `bubble_E2P`:

$$P_b \;=\; (\gamma-1)\,\frac{3}{4\pi}\,\frac{L_{\rm mech}}{v_{\rm mech}R_1^2}
\;\;\xrightarrow{\;\gamma=5/3\;}\;\;
\frac{L_{\rm mech}}{2\pi v_{\rm mech}R_1^2}\;=\;\frac{\dot p}{4\pi R_1^2},$$

so the wind force reaching the shell is $F = 4\pi R_2^2 P_b = \dot p\,(R_2/R_1)^2$.

*Two caveats.* The clean $\dot p/4\pi R_1^2$ form is exact only at the default `gamma_adia = 5/3`,
which is user-exposed. And the identity holds for the **stored** `params['Pb']` at all times, but for
the pressure the **ODE actually integrates** only outside the `dt_switchon` ramp window — for
$t \le t_{\rm SF}+10^{-3}\,$Myr the ODE uses $R_1$ scaled linearly toward zero
(`get_bubbleParams.py:495–503`). With `TFINAL_ENERGY_PHASE = 3e-3` that window is up to the first
**third** of phase 1a's maximum duration (exactly a third at the default `tSF = 0`; the phase-1a loop
bound is absolute $t$ while the ramp is $t_{\rm SF}$-relative), and your own `PLAN.md` §1(3) measures
the two pressures differing by up to $3.31\times$ inside it. Any $\alpha_p$ diagnostic built on Identity I must be
reported as invalid there rather than silently wrong.

**(II) The momentum phase applies exactly $F = \dot p$, i.e. $\alpha_p = 1$ in the force sense.**
This is *not* a consequence of Identity I. In phase 2 `bubble_E2P` is never called; the drive is built
directly from `pRam(R2)` (`run_momentum_phase.py:445,272,585,669`), and with
$v_{\rm mech}=2L_{\rm mech}/\dot p_{\rm tot}$ (`update_feedback.py:181`),
$4\pi R_2^2 P_{\rm ram} = \dot p_{\rm tot}$ identically.

`run_momentum_phase.py:587–588` (and `:893`) additionally assign `params['R1'].value = R2` with the
comment *"Set R1 = R2 (no inner shock in momentum phase)"*. **That assignment is bookkeeping, not
dynamics** — `params['R1']` is read in exactly one place in the tree, `get_bubbleParams.py:115`
inside `cool_beta_to_Ebdot`, which is energy/implicit-phase code. So it sets the output column and
states the modelling assumption; it does not force anything.

The assumption it states is nonetheless the right one, and Lancaster say the same from the other
side: *"in the limit that $\alpha_p \to 1$ we have $\mathcal{R}_f \to \mathcal{R}_w$ and thus that,
in the idealized, purely momentum-driven solution the entire bubble is made up of the free wind
region"* (`eq:alphap_shock`). So TRINITY's momentum phase **is** the $\alpha_p = 1$ idealised
momentum-driven limit — not a missing factor. See §4.4.

---

## 2. Where you sit relative to each paper

### 2.1 Lancaster Paper I — the direct comparison

Paper I is analytic; it runs no simulations of its own (the 3D runs are Paper II's). It criticises
WARPFIELD-class models in at least nine passages, which reduce to **six** distinct criticisms — not
the three revision 1 listed. Scored against TRINITY:

| # | Criticism (labels) | TRINITY |
|---|---|---|
| 1 | **The thin-shell lumping itself.** Feedback channels "combine many feedback channels artificially into a single thin-shell evolution equation… This approximation is made in order to simplify calculations but does not have rigorous theoretical foundations" (`sec:intro`); "The model assumes that the bubble is made up of a single thin-shell where all forces… are applied" (`subsec:theory_review`); it "prevent\[s\] a faithful representation of the gas density and ionization structure" (`subsec:problems`) | **open** — this is Paper I's central thesis, and §4.1/§4.3 are the answers to it. Partial credit on the third statement: your shell solver *does* resolve the internal structure; it does not separate $\mathcal{R}_w$ from $\mathcal{R}_i$ |
| 2 | **No turbulently enhanced cooling.** "does not include a model for turbulently enhanced cooling… which could make the WBBs act in a more momentum-driven manner much earlier"(`subsec:theory_review`); conduction-only, spherical heat dissipation misses the dominant loss channel (`subsec:problems`) | **open**; this is what `cooling_boost_fA` / `cooling_boost_kappa` attack, and TRINITY has the architecture — see below |
| 3 | **The photoionised thermal-pressure force is omitted from the momentum equation** (`subsec:theory_review`) | **structurally fixed** — see below |
| 4 | **Constant-density PIR / background inhomogeneity.** "these semi-analytic models must take into account inhomogeneities in the background medium into which they expand" (`subsec:problems`); "Principal among these would be to remove the assumption of a constant density PIR" (`sec:conclusion`) | **open** — and §5's $f_{\rm ion}$ knob is the cheap 1D proxy |
| 5 | **The early phase in which WBB and PIR evolve independently** is "distinct from simpler models… which use a single thin-shell approximation" (`subec:early_evol` — the paper's own typo) | **open** — needs $\mathcal{R}_w\ne\mathcal{R}_i$ (§4.3). Note the paper says "distinct from", not "cannot represent" |
| 6 | **LyC "trapping" is assumed.** It is "convenient for theoretical modeling… \[but\] is often not seen in simulations with a cluster of massive stars… or uniform density backgrounds" (`app:trapping`) | **partly answered** — you compute trapping rather than assuming it (§3.5); the geometry that produces it is still 1D. See §5 |

A seventh criticism in the same section — that instantaneous thermal/ram-pressure balance "ignores the
inertia of the shell" (`subsec:theory_review`, citing Raga et al. 2012b) — is aimed at the
pressure-balance class, which TRINITY is not in; it is used in §2.4 against Geen 2019 instead.

**On (4).** `include_PHII` + `get_phii_c3c` put a photoionised term into $P_{\rm drive}$ in all four
phases, and in the transition and momentum phases it dominates in every configuration measured. In
1a/1b the confined branch fires and $P_{\rm HII}$ is exactly `0.0`, so $P_{\rm drive}=P_b$ there —
**but that is a property of the regime, not a theorem.** Batch 7's confinement screen records
`B3MW001` ($L_w\times0.01$) at **78.4% HII-dominated in the energy phase**, with `ratio_max` = 4.927
(`data/b7_confinement_screen.csv`; the run is `run_complete = False` and never reaches transition or
momentum, so PLAN.md marks it VOID for any *driving-branch* claim — it is evidence about 1a/1b only).
of all five configs" was a stale Batch-5 read and is withdrawn.

**On (6).** Paper I's "Prospects" sentence is easy to truncate favourably. In full:

> *"The ideal version of these semi-analytic models would include a parameterized model for this
> heat dissipation that is solved on-the-fly and included in the WBB energy equation… This is done
> for the case of only conductive heat dissipation in a spherical scenario by the* `WARPFIELD`
> *models, **but as \citet{Lancaster21a,Lancaster21c} has shown, cooling in turbulently-mixed
> intermediate-temperature gas can certainly dominate energy losses.**"*

So TRINITY has the *architecture* Paper I asks for and half the *physics*. That framing is
defensible; the sentence must be quoted past the comma.

**The C3c switch point is Lancaster's $R_{\rm ch}$ — exactly, and only in the momentum phase.**
Setting $P_{\rm C3a}(R_2)=P_{\rm conf}(R_2)$ with $P_{\rm conf}=P_{\rm ram}$:

$$\frac{\dot p}{4\pi R_2^2}=\bar\rho c_i^2\!\left(\frac{R_{\rm St}}{R_2}\right)^{3/2}
\;\;\Longleftrightarrow\;\;
R_2=\frac{R_{\rm eq}^4}{R_{\rm St}^3}\;=\;R_{\rm ch},$$

using $R_{\rm eq}^2 \equiv \alpha_p\dot p/(4\pi\bar\rho c_i^2)$ (**`eq:RE_MD_def`**, restated in
`eq:ReqMD_app`; note `eq:eta_def` defines $\zeta$ itself, not $R_{\rm eq}$) and the relation
$R_{\rm ch}=R_{\rm eq}^4/R_{\rm St}^3$ (**`eq:Rch_Req_rel`**, restated dimensionally in
`eq:Rch_app`; the *definition* of $R_{\rm ch}$ is `eq:Rch_def`,
$R_{\rm ch}\equiv\frac{\alpha_B}{12\pi(\mu_Hm_Hc_i^2)^2}\frac{\alpha_p^2\dot p^2}{Q_0}$). SymPy
returns crossover$/R_{\rm ch} = 1$ exactly **at $\alpha_p=1$**; since $R_{\rm ch}\propto\alpha_p^2$,
at any other $\alpha_p$ the crossover sits at $R_{\rm ch}/\alpha_p^2$. That is a consistency check
rather than a caveat: TRINITY's phase-2 confining pressure $\dot p/(4\pi R_2^2)$ *is* the
$\alpha_p=1$ wind pressure, so $\alpha_p=1$ is the right value to evaluate $R_{\rm ch}$ at. So in
phase 2 your confined branch is precisely Lancaster's wind-dominated $R_2<R_{\rm ch}$ and your
unconfined branch their PIR-dominated $R_2>R_{\rm ch}$.

**Three caveats.**

- `get_phii_c3c` compares against `params['Pb']`, which is $P_{\rm ram}$ **only in the momentum
  phase**. In 1a/1b/1c it is the bubble *thermal* pressure — larger by $(R_2/R_1)^2$ — so there the
  switch is the energy-driven analogue of $R_{\rm ch}$, the same force-balance *structure* but not
  the quantity Lancaster tabulate (they define $R_{\rm ch}$ for the MD case). A natural
  generalisation, not an error — but `t_cross` in your ladder is **not** the $R_2=R_{\rm ch}$
  crossing.
- The reduction assumes $f_{\rm abs,ion}=1$. Since $P_{\rm C3a}\propto\sqrt{f_{\rm abs}}$ and
  $P_{\rm ram}\propto R_2^{-2}$ while $P_{\rm C3a}\propto R_2^{-3/2}$, the crossover scales as
  $R_2^{\rm cross}\propto f_{\rm abs}^{-1}$ — **partial absorption pushes the crossover outward**,
  it does not shrink it. (Revision 1 said $f_{\rm abs}^{1/3}$, which is wrong in both exponent and
  sign.) In practice this rarely bites: B11.0 measures $f_{\rm abs}=1.0000$ on 16/16 transition and
  13/17 momentum driving rows at nominal wind, and on **all** driving rows at $L_w\times0.1$.
- $R_{\rm ch}$ is proportional to $\alpha_p^2$, so quoting a $R_2/R_{\rm ch}$ requires stating which
  $\alpha_p$ was used.

With those caveats the correspondence stands and is worth stating in the method paper. Batch 8's
result — C3a reproduces Hosokawa–Inutsuka to 0.0000% over $R/R_{\rm St}\in[2,50]$ — is the same
statement seen from the Spitzer side, and should be quoted with the caveat PLAN.md attaches to it:
*"Not independent confirmation — HI is derived from the same momentum equation, so once the algebra
gates hold the ODE must return HI."* (Its G8.4 gate failed as registered, 9.511% against a 5% bar,
and was amended.)

**Where TRINITY is better than the CEM:** the CEM ignores gravity, external pressure, direct and
indirect radiation pressure, the cloud density profile, and time-variable feedback. It *does* solve
an energy equation in its ED variant (`subsec:ed_jfb`), so the honest count is **six**, not seven,
and what the ED-CEM lacks is a self-consistent dissipation model, not the equation.

**Where the CEM is better:** it separates $\mathcal{R}_w$ from $\mathcal{R}_i$. TRINITY's $R_2$ does
both jobs. That is the root of §4.1 and §4.3.

### 2.2 Lancaster Paper II — the calibration, and what it is actually a calibration of

Paper II's `tab:cem_comp` reports $\langle\alpha_p\rangle$ = 4.64 / 4.78 / 4.57 (HWR at
$N=128/256/512$) and 5.57 / 6.20 / 6.82 (MWR). **The MWR values rise monotonically with resolution and
the paper reports the trend without claiming convergence**: *"For the* `WRM` *simulations, we see a
tendency to larger $\alpha_p$ with higher resolution."* (It also notes *"there is no clear pattern
towards smaller values of $\langle\Delta\rangle$ with resolution"*.) Quote the range, not a single
number.

Crucially, **these $\alpha_p$ are measured from `eq:alphap_derive`**,
$\alpha_p = \tfrac34\frac{\mathcal{V}_w/4}{\langle v_{\rm out}\rangle}\frac{4\pi\mathcal{R}_w^2}{A_w}$,
which is a product of **two** independent factors, and Paper II names them as such: *"the dynamics of
WBBs are affected by properties of cooling at their interfaces in two distinct ways: dissipation and
geometry."*

- the **dissipation** factor $\tfrac{3}{16}\mathcal{V}_w/\langle v_{\rm out}\rangle$ — how much
  energy the interface loses;
- the **geometry** factor $4\pi\mathcal{R}_w^2/A_w$ — the spherical-to-fractal area ratio.

**Only the second is unavailable in 1D**, where $A_w\equiv4\pi\mathcal{R}_w^2$ identically. The first
is a physics parameter a 1D code can and does carry — TRINITY's is the bubble energy budget, and it
is why $(R_2/R_1)^2\gg1$ in the energy phase. So the correct statement is *not* "1D forbids
$\alpha_p>1$" (revision 1's error, in the opposite direction from its first one) but: **1D forbids
the geometric half of the enhancement, and $A_w$ is exactly what `cooling_boost_fA` parameterises on
the energy side.** Paper II is in any case explicit that the values are resolution- and
geometry-dependent: *"The exact values of $A_w$ and $\langle v_{\rm out}\rangle$ (and therefore
$\alpha_p$) in reality will likely be different from those in our simulation."*

**$(R_2/R_1)^2$ *is* the directly comparable quantity — revision 1 said the opposite and was wrong.**
Lancaster take the hot-gas pressure at the post-shock value
$P_{\rm hot} = 3\dot p/(16\pi\mathcal{R}_f^2)$, while TRINITY's `get_r1` uses the free-wind *ram*
value $\dot p/(4\pi R_1^2)$. At the same pressure $\mathcal{R}_f = (\sqrt3/2)R_1$, so
$x \equiv \mathcal{R}_w/\mathcal{R}_f = (2/\sqrt3)(R_2/R_1)$, and substituting into
`eq:alphap_shock`, $\alpha_p=\tfrac14[3x^2+x^{-2}]$:

$$\boxed{\;\alpha_p \;=\; \left(\frac{R_2}{R_1}\right)^{2} + \frac{3}{16}\left(\frac{R_1}{R_2}\right)^{2}.\;}$$

The $4/3$ convention mismatch cancels the $3/4$ in the leading term **exactly**. The residual
$\tfrac{3}{16}(R_1/R_2)^2$ is $18.75\%$ of the leading term at $R_2/R_1=1$, $1.2\%$ at 2, $0.2\%$
at 3, negligible above.

**Two things must be said about this mapping, or it misleads.** First, it comes from matching the two
conventions *at the same bubble pressure* — which is the right invariant, because the bubble pressure
is what exerts the force on the shell, and $R_1$ and $\mathcal{R}_f$ denote the same physical surface
(the wind termination shock) computed under two shock conditions, TRINITY's the ram-pressure form and
Lancaster's the exact strong-shock post-shock value. Identifying $\mathcal{R}_f\equiv R_1$
*geometrically* instead would give $\tfrac34(R_2/R_1)^2+\tfrac14(R_1/R_2)^2$ and no cancellation;
the pressure-matched version is the one to use.

Second, **$\alpha_p$ and the force ratio are not the same quantity.** Lancaster's own `eq:Phot_EC`
gives the exact force as $4\pi\mathcal{R}_w^2 P_{\rm hot} = \tfrac34\dot p\,x^2$, and writes
$\approx\alpha_p\dot p$ only *"using the assumption $\alpha_p \gtrsim 1$"* — i.e. dropping the
$x^{-2}$ term. Under the pressure match, $\tfrac34 x^2 = (R_2/R_1)^2$ exactly. So:

$$\underbrace{(R_2/R_1)^2}_{\text{TRINITY's exact force}/\dot p}
\;=\;\underbrace{\tfrac34 x^2}_{\text{Lancaster's exact force}/\dot p}
\;\le\;\underbrace{\tfrac14[3x^2+x^{-2}]}_{\alpha_p,\ \text{eq:alphap\_shock}} .$$

**Emit $(R_2/R_1)^2$.** It is TRINITY's exact force ratio *and* Lancaster's exact force ratio, and it
is the quantity Paper II's tabulated $\alpha_p$ approximates. Adding the $\tfrac{3}{16}(R_1/R_2)^2$
term reproduces `eq:alphap_shock` exactly but is actively misleading where $R_2\to R_1$: it would
report $1.1875$ for a bubble whose force ratio is exactly 1.

One further limit on the comparison: `eq:alphap_shock` is flagged as true *"excluding geometric
effects which are treated more carefully in Appendix A of \citet{Lancaster24a}"*, so a 1D
$(R_2/R_1)^2$ maps onto the **spherical-equivalent** $\alpha_p$ only. Paper II itself performs
exactly this inversion for the literature in `app:sim_comp`, so the procedure is theirs, not an
invention.

**Photoionisation raises $\alpha_p$** — 2.55 → 4.66 (HD), 4.09 → 6.20 (MHD) — by removing
collisional Ly$\alpha$ as an interface coolant and by smoothing background inhomogeneity so $A_w$
drops. Note the sign: adding photoionisation makes the *wind* more effective. One caveat worth
carrying: Paper II `subsec:cooling` documents that the wind-only baseline suffers numerical
diffusion of neutral H into $10^4$–$10^{5.5}$ K gas, *"leading to an excess of cooling due to
Ly$\alpha$ that otherwise would not be present in this gas"*. The authors argue this *"does not detract from any
of the main results"* and treat the increase as physical; they do **not** attribute any part of
2.55 → 4.66 to the artefact. Cite the increase as their result, and cite the caveat as theirs too.

### 2.3 Geen & de Koter 2022 — the same shell physics, with the chronology the right way round

`get_shellODE.py` implements Geen 2022's `eqn:draine1`–`eqn:draine3` — the Draine (2011) hydrostatic
dusty-ionised-shell system with $\phi$ and $\tau$ integrated outward — with the same inner boundary
condition $P_w = P_i \equiv (m_H/X)n_ic_i^2$ (`eqn:PwPibalance`; TRINITY `shell_structure.py:125-126`).
Geen 2019 closes the same system identically (`wind:windpressurebalance`).

**This is a common inheritance, not TRINITY implementing Geen.** Geen 2022 routes *its own* equations through
Draine (2011) and Martínez-González et al. (2014); Paper I attributes WARPFIELD's hydrostatic shell
to Abel+2005 / Pellegrini+2007 / Draine 2011 / Kim+2016, so the shared ancestor is Draine (2011), not
the Martínez-González line. Geen 2022 then cites `Rahner2017` — WARPFIELD, TRINITY's parent — in the
list of models that already apply this analysis with an embedded wind bubble. Revision 1's *"you
already implement their model"* inverted a chronology in which TRINITY's lineage is the *earlier*
work. The correct statement is that the two agree term-for-term, which is a mutual corroboration.

**What TRINITY does that they do not:** solve it self-consistently along a dynamical trajectory. Geen
2022 solve it on a prescribed $r_w(t)$ from their analytic Weaver-like solution (`eqn:rwt`).

**What they do that TRINITY does not** — and revision 1's *"you are strictly ahead here"* is
withdrawn:

- **Metallicity-dependent dust.** $\sigma_d = 10^{-21}\,\mathrm{cm}^{-2}\,Z/Z_\odot$, explicitly.
- **Star-dependent $T_i$.** Tabulated per stellar model (`table:app_starprops`), not pinned to $10^4$ K.
- **$\Omega < 4\pi$ solid-angle geometry.** (`coverFraction` is an energy leak, not a solid angle —
  different physics.)
- **An analytic overflow radius**, `eqn:overflowcondition`, with the clean $\omega$ vs $5/4$
  threshold: for $\omega<5/4$ overflow becomes *less* likely with radius (trapping wins), for
  $\omega>5/4$ *more* likely. `densPL_alpha` spans both sides, so this is a sharp, cheap validation
  target.

The first two are exactly what §4.5 charges TRINITY with lacking.

**The overflow criteria are not the same criterion.** Geen's *numerical* test is
$M_i \ge M(<r_i)$ with $M_i=\int_{r_w}^{r_i}\Omega r^2n_i(m_H/X)\,dr$ (`eqn:photoionised_mass`);
their *analytic* version adds *"a further simplifying assumption that the mass swept up by the shell
$M(<r_i)\simeq M(<r_w)$."* TRINITY's loop
(`shell_structure.py:158,182`) runs `while not is_allMassSwept and not is_phiDepleted` against
`mShell_end = params['shell_mass']` (`:107`), integrating outward from `rShell0 = R2` (`:106`).
`shell_mass` is set from `get_mass_profile(R2, params)` immediately before each shell solve in every
runner, i.e. it is the mass originally interior to $R_2$ — $M(<r_w)$ in Geen's notation. (B11.0's
`shell_mass/M_avail` = 0.999997–1.000000 while $R_2$ grows $4.65\times$ is independent confirmation:
only $M(<R_2)$ can saturate that way.) **So TRINITY implements Geen's *analytic* criterion, not their numerical one**, and
compares against the smaller reference mass, so it overflows *earlier*. The discrepancy is
$M(<r_i)/M(<r_w) = (r_i/r_w)^{3-\omega}$; Geen 2022 Fig. `fig:ionisedshellthickness` shows
$(r_i-r_w)/r_w$ approaching unity near overflow, so at $\omega=2$ this is a factor up to $\sim2$ in
mass. Worth a one-line comparison in the paper; not obviously wrong, but it is a choice you are
currently making implicitly.

### 2.4 Geen et al. 2019 — you are better on the one thing that matters

Geen 2019's dynamical closure is a *ram-pressure balance* at the front,
$n_ic_i^2 = n(r_i)(\dot r_i+v_0)^2$ (`photo:externalpressure`), with **no shell inertia**. Lancaster
Paper I `subsec:theory_review` flags instantaneous force balance as the key weakness of this class
(citing Raga+2012b). **TRINITY integrates the full thin-shell momentum equation including
$\dot M_{\rm sh}v_2$ and $M_{\rm sh}\dot v_2$** — Hosokawa–Inutsuka rather than Spitzer. That is the
right call and you should say so.

Two things to take from Geen 2019:

- **It licenses your transparent cavity.** *"Since the temperature of this gas is typically well
  above the limit to be collisionally ionised, the UV photons from the star are not absorbed by the
  wind bubble"* (`winds_in_uv`). TRINITY's `phi0 = 1` at `shell_structure.py:119-120` is therefore
  the standard picture and is **correct**. What is not standard is the balance *volume* (§4.1).
- **A cross-check quantity.** $C_w = 2^{1/4}\left(\frac{\dot p_w}{4\pi}\frac{X}{m_H}\frac{1}{c_i^2}\right)^{3/2}\left(\frac{Q_H}{\alpha_B}\frac{3}{4\pi}r_i\right)^{-3/4}$
  (`wind:coefficient`), $\approx0.0093$ at their fiducial values, with $C_w>1$ meaning wind-dominated.
  Same ordering as Lancaster's $\zeta$ but different exponents; computing both from TRINITY output
  would be a cheap, high-value figure.

Their cavity-corrected balance $\tfrac{4\pi}{3}n_i^2(r_i^3-r_w^3)\alpha_B=Q_H$
(`wind:photoequilibrium`) is one of the three published sources behind §4.1.

### 2.5 Haid et al. 2018 — the sanity anchor. There is no tension.

Haid's headline is that the **ambient medium** decides the winner, and that the dependence on it is
stronger than on the source: radiation dominates by $\sim\!50\times$ in CNM ($n_0=100$), winds
dominate by $10^2$–$10^4$ in WIM ($n_0=0.1$, $T=10^4$ K), with the switch near
$n_0\sim1\,$cm$^{-3}$. Their $\bar T_{\rm HII}$ = 7160–8150 K spans *both* stellar mass and ambient
phase, and they use $\alpha_B = 2.56\times10^{-13}(T/10^4)^{-0.83}$.

They also measure non-additivity directly:

> *"In the CNM, the feedback from both processes $p_{\rm Combi}$ is larger than
> $p_{\rm IRad}+p_{\rm Wind}$ by $\sim$ 1, 3, and 23 percent… In the WIM, the difference is a factor
> of $\sim$ 3.2, 2.8, and 1.9."*

**Revision 1 claimed this contradicts Lancaster. It does not — I had misread Paper I.** Paper I
`eq:force_low_eta` gives, in the *low*-$\zeta$ limit,
$\dfrac{4\pi\rho_i\mathcal{R}_i^2}{4\pi\bar\rho R_{\rm St}^2}\approx1+\tfrac12\zeta^3$, and the
surrounding text is unambiguous: *"at small values of $\mathcal{R}_{\rm eq}/R_{\rm St}$ the momentum
from the CEM solution is actually **larger** than the momentum given by the separate idealized
solutions."* The mechanism they give — *"the wind bubble provides a volume of gas that is already
ionized, liberating LyC photons to ionize gas further out"* — is a super-additive coupling, the same
direction Haid measure. And Paper II `subsubsec:ideal_sim_review` places Haid's CNM
runs at $\mathcal{R}_{\rm eq}/R_{\rm St} = 0.35$ *"(all other values are smaller)"* — squarely on
that branch. (`app:sim_comp` is where the input parameters and the assumed
$\alpha_p = 10,\,3,\,1$ for the 12 / 23 / 60 $M_\odot$ cases live.)

The $\sim$35% over-estimate is stated for a **different variable**: Paper I `subsec:md_jfb` gives it
at $\mathcal{R}_i/R_{\rm ch}\approx1$, and since $R_{\rm ch}=R_{\rm eq}^4/R_{\rm St}^3$ that is not
the same condition as $\zeta\approx1$. But do not read that as "over-estimation belongs only to the
$R_{\rm ch}$ axis": immediately after `eq:force_low_eta` Paper I also says *"it is clear from
[`fig:numerical_solution`] that the opposite is true at near-unity values of
$\mathcal{R}_{\rm eq}/R_{\rm St}$"* — i.e. the sum over-predicts at $\zeta\approx1$ too. The
sign change is between *low* $\zeta$ and *near-unity* $\zeta$. Paper I's global statement is *"the
momentum evolution of the joint feedback bubble is still within 25% of the naive value given by the
sum of the idealized solutions in all models presented."*

**Net, and this is the usable version:** the naive sum is a **lower** bound at low $\zeta$ (Haid's
regime, and Paper I agrees) and an **upper** bound near $\mathcal{R}_i\approx R_{\rm ch}$ (by
$\sim$35%), crossing between. That sign change with regime is precisely why a coupled closure beats
choosing `max` or `+` globally — see §4.3.

*Also withdrawn:* revision 1 said Paper II `subsec:PIR_cooling_effect` "independently confirms Haid's
direction". It does not measure $p_{\rm Combi}$ vs $p_{\rm IRad}+p_{\rm Wind}$ at all; it measures
$\alpha_p$ rising when LyC is on. Related in spirit, different quantity.

---

## 3. What is unambiguously right

1. **Shell momentum equation.** $\mathrm{d}(Mv)/\mathrm{d}t$ form with the $\dot M v$ term. Correct,
   and better than Geen 2019's instantaneous pressure-balance closure.
2. **$R_1$ from free-wind ram balance.** The classical Weaver (1977) inner-shock condition
   $\rho v^2 = P_b$. *Flagged as outside the verified corpus:* Weaver 1977 is not one of the five
   papers read, and none of them states it — the only in-corpus form is Lancaster's exact
   strong-shock $P_{\rm hot}=3\dot p/(16\pi\mathcal{R}_f^2)$, which is $3/4$ of TRINITY's at the
   same radius. Both conventions are in use; just don't mix them when quoting $\alpha_p$ — see §2.2
   for the exact conversion.
3. **The transparent cavity.** `phi0 = 1` is the standard picture, licensed verbatim by Geen 2019
   `winds_in_uv`.
4. **Continuity of the energy→momentum handover.** Because $P_b\to\dot p/(4\pi R_1^2)$ and
   $R_1\to R_2$ as $E_b\to0$, $P_b\to P_{\rm ram}$ automatically, so `max(P_thermal, P_ram)` in the
   transition phase is a genuine continuous handover, not a patch. Your seam ratios 0.995–0.999
   confirm it **for the transition→momentum seam specifically**; `PLAN.md` §3c.1 records the other
   seams as 0.89–0.92 (energy→implicit), 0.53–0.96 (implicit→transition) and 0.86–0.99 (regime
   switch), so quote the 0.995–0.999 with its seam attached.
5. **Self-consistent radiation trapping.** You compute $f_{\rm esc,ion}$ from the actual shell
   structure instead of Lancaster's threshold-density formula (`eq:phot_trap_rhobar`) or Geen 2022's
   prescribed $r_w(t)$. This directly addresses Paper I `app:trapping`, which criticises trapping as
   an *assumption*.
6. **The confined branch is exactly self-consistent** — your own §6b establishes this and I agree:
   hot transparent cavity, skin in pressure equilibrium at $P_b$, drive $=P_b$ transmitted, no cavity
   gas mass claimed. Every book balances. It is the driving branch that does not.
7. **$P_{\rm ext}$ and $P_{\rm ISM}$.** Lancaster Paper I explicitly drops external pressure
   (`sec:theory_joint`: *"we ignore … any external pressures or the effects of gravity"*). You keep
   both, with the right sign. (Paper II does carry a background; the omission is Paper I's.)
8. **The energy bookkeeping of the photoionised term.** The energy ODE charges `press_bubble·dV`
   only (`energy_phase_ODEs.py:274`), so photoionised work is not drawn from $E_b$. Correct — its
   source is the radiation field, continuously resupplied.

---

## 4. What can be better — ranked by *measured* impact

### 4.1 (highest) Balance recombination over the ionised layer, not the wind cavity

`get_phii_c3c` sets $n_{\rm C3a}$ from $Q_{i,\rm abs} = \tfrac{4\pi}{3}R_2^3\,\chi_e\alpha_Bn^2$ —
a Strömgren balance over the **cavity**. Three papers and TRINITY's own shell solver put the
photoionised gas between $r_w$ and $r_i$ and balance over $r_i^3-r_w^3$ (Geen 2019 and Geen 2022
share an author, and Geen 2022's uniform form is the analytic reduction of its own numerical
`eqn:photoionisation_equilibrium` — so call it three sources plus your code, not four independent
ones):

| source | expression |
|---|---|
| Lancaster Paper I `eq:ionreceq2` | $\tfrac{4\pi}{3}(\mathcal{R}_i^3-\mathcal{R}_w^3)\alpha_Bn_{\rm Hi}^2=Q_0$ |
| Geen 2019 `wind:photoequilibrium` | $\tfrac{4\pi}{3}n_i^2(r_i^3-r_w^3)\alpha_B=Q_H$ |
| Geen 2022 `eqn:photoionisation_equilibrium_uniform` | $Q_H=\tfrac{4\pi}{3}n_i^2(r_i^3-r_w^3)\alpha_B$ |
| **TRINITY** `shell_structure.py:243` | `_vol_ion = R_IF**3 - rShell0**3` |

C3a balances over $\tfrac{4\pi}{3}R_2^3$, which in TRINITY's geometry is $\tfrac{4\pi}{3}r_w^3$ —
**exactly the volume all four exclude.** Lancaster add the physical reason: *"the WBB enhances
$\rho_i$ relative to the classical solution, due to the presence of $\mathcal{R}_w$ in the
denominator."*

The reason to put this first is that it is the *same* defect your Batch 11 measures as seam C. A
cavity Strömgren-filled at $n_{\rm C3a}$ implies $M_{\rm cav}/M_{\rm shell}$ = 0.0952 → **0.5638** by
$t=1.5$ on B3M — 57,397 vs 101,805 $M_\odot$, reproduced by three independent routes — while
`shell_mass` already equals **100.0000%** of the gas the run has and the winds inject 54.8 $M_\odot$.
On the published picture that gas should not exist at all.

**Two things I would carry from your own measurements into any write-up.** First, B12 showed the
0.5638 is regime-scoped: $M_{\rm cav}\propto R_2^{3/2}\sqrt{Q_if_{\rm abs}}$, so the seam tracks
**bubble size**, not the degree of HII dominance, and $L_w\times0.1$ gives 0.1296. Always quote it
with the config. Second, B11.A's degeneracy result is the strongest argument for a coupled closure
rather than a patch: the photon-conserving fixed point $x = f_{\rm abs}(Q_i(1-x))$ has the unique
root $x=1$ on 33/33 driving rows, i.e. C3a's own scheme cannot be made photon-conserving without a
second equation. Both Geen 2019 `wind:windpressurebalance` and Geen 2022 `eqn:PwPibalance` supply
exactly that second equation, and it is the one your shell solver already uses at
`shell_structure.py:125-126`. That is your K5/K6 pair, and I would take K6.

### 4.2 Shell inertia should exclude the ionised interior — and this is first-order *now*

TRINITY puts the entire swept mass inside $R_2$ into the shell. Lancaster Paper I
`eq:pr_spitzer_adj` subtracts the ionised gas:

> *"A simple adjustment to this momentum inference is to use the shell mass
> $M_{\rm sh} = 4\pi\mathcal{R}_i^3(\bar\rho-\rho_i)/3$, that is, to subtract out the mass in ionized
> gas. **Though this reduction is not consistent with the derivation of** `eq:HIImomentum2`, we will
> see that it can be more accurate."*

Paper II validates it empirically (`app:spitzer_momentum`, `fig:sptiz_momentum_comp`) and drops the
caveat. **Carry the caveat.** The adjustment is not a consistent re-derivation — the solution
$\mathcal{R}_{\rm Sp}(t)$ is not re-derived under the adjusted mass — so implement it as a debit on
the inertia with the inconsistency stated, exactly as Paper I does.

**Revision 1 called this "currently near-irrelevant (your ionised layer is a thin skin)". That is
false and your own B11 measured it false.** On B3M's momentum rows $dR_{\rm ion}/R_2$ = 0.6579–1.3076
with $dR_{\rm ion}/dR_{\rm full}$ median **0.9954** — the momentum shell is essentially *entirely*
ionised. B11.C2 measured the cost: $R_2(t=1.5)$ moves **+8.55%** (inertia only) or **+9.22%**
(inertia and gravity), against an offline control reproducing the run to 0.871%. At $L_w\times0.1$
it falls to +0.45% / +0.97%, tracking $M_{\rm cav}$.

So this belongs high in the ordering, not "the moment §4.3 lands".

### 4.3 The composition rule: `max` / `+` vs a single coupled closure

**Your open question**, from `get_phii_c3c`'s docstring: *"$P_{\rm C3a}\propto R_2^{-3/2}$ vs
$P_{\rm ram}\propto R_2^{-2}$: does a real momentum-phase cavity stay Strömgren-filled?"*

**Lancaster answer it for the momentum-driven case.** In the co-evolution phase the ionised gas is
in pressure balance with the wind bubble, so the pressure at the shell is *always* the wind-bubble
pressure $\alpha_p\dot p/(4\pi\mathcal{R}_w^2)$ — photoionisation never adds an independent
pressure. What it changes is **where** that pressure acts: `eq:ionreceq2` + `eq:RiRw_rel` give
$\mathcal{R}_i=\mathcal{R}_w(1+\mathcal{R}_w/R_{\rm ch})^{1/3}$, hence (from
`eq:HIImomentum_joint1`)

$$F_b=4\pi\mathcal{R}_i^2 P_i=\alpha_p\dot p\left(\frac{\mathcal{R}_i}{\mathcal{R}_w}\right)^{\!2}
=\alpha_p\dot p\left(1+\frac{\mathcal{R}_w}{R_{\rm ch}}\right)^{\!2/3}.$$

**Scope, which revision 1 got wrong.** This closure is derived in `subsec:md_jfb` for the
**momentum-driven phase only**. The energy-driven variant (`subsec:ed_jfb`) replaces
$P_i=P_{\rm hot}$ and requires a separate energy equation — four equations in $\mathcal{R}_w$,
$\mathcal{R}_i$, $E_w$, $P_{\rm hot}$. So the numbers below compare **composition rules against the
MD-phase CEM**; they are *not* a phase-by-phase error budget for TRINITY, and labelling a
$-33\%$ column "1a/1b" (as revision 1 did) is a category error.

Also worth stating plainly: Paper I's pressure balance is **imposed, not derived** — *"We now imagine
that the WBB has come into equilibrium with the PIR with no net force being applied across its outer
edges"*, and Paper II: *"the WBB and the PIR are **forced** to be at the same pressure."* It is
enforced instantaneously at $t_{\rm eq}$, which requires one state variable ($\mathcal{R}_i$) to jump.

I verified numerically that the MD closure has the right limits:

- $\to \alpha_p\dot p$ (your **confined** branch) as $\mathcal{R}_w/R_{\rm ch}\to0$ — 1.0007 at $10^{-3}$;
- $\to F_{b,\rm Sp}$ (your **unconfined** branch) as $\mathcal{R}_w/R_{\rm ch}\to\infty$ — 1.00007 at $10^4$;

so your two branches *are* the correct asymptotes of the MD CEM. Paper I is careful that the
strong-wind limit `eq:approxF2` is stated for $\mathcal{R}_w\ll R_{\rm ch}$ and the weak-wind limit
`eq:approxF1` for $\mathcal{R}_w\gg R_{\rm ch}$; do not claim exactness outside those.

Composition rules against the MD CEM, at TRINITY's convention $R_2\equiv\mathcal{R}_i$:

| $\mathcal{R}_i/R_{\rm ch}$ | $F_{\rm sum}/F_{\rm CEM}$ | $F_{\max}/F_{\rm CEM}$ |
|---|---|---|
| 0.5 | 1.337 | 0.783 |
| **1.0** | **1.342** | **0.671** |
| 2.0 | 1.318 | 0.772 |
| 10 | 1.209 | 0.918 |
| 100 | 1.083 | 0.984 |

The 1.342 at $\mathcal{R}_i=R_{\rm ch}$ is an independent reproduction of Lancaster's stated
$\sim$35% (`fig:force_comp`), which is the check that my normalisation is right.

**What this licenses you to say.** In the **momentum** phase, where TRINITY uses the bare sum and
Lancaster's MD closure applies, $P_{\rm HII}+P_{\rm ram}$ over-estimates the coupled force by up to
$\sim$34%, worst at $\mathcal{R}_i\approx R_{\rm ch}$ — which is exactly where your C3c switch sits.
Your own measurement of the same defect from the other side is that the `+P_ram` term is a 14.1%
double-count on B3M momentum rows (`P_drive/P_ram` median 7.095 → 6.095 under transmit). The `max`
column applies to energy-driven phases where the MD CEM does *not* apply, so treat it as an
indication that the two rules bias in opposite directions, not as a quantified error.

Also worth knowing, weak-wind limit (`eq:approxF1`):
$F_b\approx F_{b,\rm Sp}+\tfrac{\alpha_p\dot p}{2}\frac{\mathcal{R}_w}{\mathcal{R}_i}$ — the wind
contributes *half* its momentum flux, further reduced by $\mathcal{R}_w/\mathcal{R}_i$. Your momentum
phase credits it the full $\dot p$ at the shell radius.

**Implementation.** You already have the pattern — `solve_R1` brackets a scalar root on $[0,R_2]$:

```python
# given R2 (= R_i, the shell), solve R2 = Rw*(1 + Rw/Rch)**(1/3) for Rw in (0, R2]
# Rch = alpha_B/(12*pi*(mu_H*m_H*ci**2)**2) * (alpha_p*pdot)**2 / Q_ion
#   NOTE: Rch propto alpha_p**2, and Q_ion here must be the ionising rate the
#   *cavity* actually receives (Qi*f_abs), not the raw SPS Q0.
Rw = brentq(lambda x: x*(1.0 + x/Rch)**(1.0/3.0) - R2, 0.0, R2)
P_drive = alpha_p * pdot / (4*np.pi*Rw**2)
```

This is a sketch, not a drop-in. It is **not** backward-compatible, and the edit surface is larger
than the Table-1 row count suggests: there are eight $P_{\rm drive}$ expression sites, of which only
**three are live** (`energy_phase_ODEs.py:253,256` and `run_momentum_phase.py:445`) and the rest are
reporting paths that must be kept in step or the force-budget output silently diverges from the
integrated dynamics. In the energy phase you would also keep $P_b$ from the bubble solve in place of
$\alpha_p\dot p/(4\pi\mathcal{R}_w^2)$ — the ED analogue, which needs the energy equation Paper I
`subsec:ed_jfb` specifies. Treat it as the shape of K6, and gate it as you gated
C3c.

### 4.4 $\alpha_p$: emit it as a diagnostic; do **not** add it as a knob

**Revision 1's §4.1 was wrong and is withdrawn in full.** It claimed $\alpha_p\equiv1$ is "almost
certainly why your momentum phase is universally HII-dominated". Four things kill that:

1. **$\alpha_p\equiv1$ in the momentum phase is not a hard-coded omission — it is what the phase
   asserts.** The force is $4\pi R_2^2 P_{\rm ram} = \dot p$ identically, from `pRam` plus
   $v_{\rm mech}=2L/\dot p_{\rm tot}$; `run_momentum_phase.py:587-588`'s $R_1=R_2$ ("no inner shock
   in momentum phase") states the same assumption for the output column, though it is bookkeeping
   rather than dynamics (§1, Identity II). Lancaster `eq:alphap_shock` says
   $\alpha_p\to1\Leftrightarrow\mathcal{R}_f\to\mathcal{R}_w$, "the entire bubble is made up of the
   free wind region" — the same statement.
2. **Half of the 4.57–6.82 enhancement is a 3D geometry effect that 1D cannot produce; the other
   half TRINITY already has.** `eq:alphap_derive` factorises into a dissipation term
   $\tfrac3{16}\mathcal{V}_w/\langle v_{\rm out}\rangle$ and a geometry term
   $4\pi\mathcal{R}_w^2/A_w$ (§2.2). Only the geometry term is identically 1 in 1D. So the honest
   claim is *not* "a 1D code should have $\alpha_p=1$" — TRINITY plainly has
   $\alpha_p=(R_2/R_1)^2\gg1$ in its own energy phase — but: **the geometric half is structurally
   unavailable, and the dissipative half is already modelled, by the bubble energy budget.** Paper I
   `subsubsec:ec_conditions` says $\alpha_p\approx1$ is expected in the momentum-driven limit *"if
   the correct dissipative scales were resolved"* and *"(at least when magnetic fields are not
   dynamically important)"* — conditions on resolution and physics, not on dimensionality. Any
   $\alpha_p>1$ imposed by hand in TRINITY would be a calibration of unresolved 3D interface
   geometry, and $A_w$ is exactly what `cooling_boost_fA` parameterises on the energy side, so the
   two must not be set independently.
3. **Batch 12 bounds it out.** At $L_w\times0.1$ the momentum phase is 100% HII-dominated at
   $P_{\rm HII}/P_b$ = **13.667–14.369**. Inverting that needs $\alpha_p\gtrsim14$, twice the largest
   value Paper II measures. And at $L_w\times0.1$, $\alpha_p P_{\rm ram}$ is small *whatever*
   $\alpha_p$ is. So $\alpha_p$ cannot be the explanation.
4. **It would not even help much at nominal wind.** Because `params['Pb']` *is* $P_{\rm ram}$ in
   phase 2, applying $\alpha_p$ raises $P_{\rm conf}$ too. With the measured $P_{\rm HII}/P_b$ =
   5.091–7.156 (median 6.165) and $\alpha_p=6.2$, roughly half the rows flip to the confined branch,
   where $P_{\rm drive}=\alpha_p P_{\rm ram}=6.2P_{\rm ram}$ against the shipped $7.095P_{\rm ram}$ —
   a ~13% reduction, not a factor, and a hard branch flip rather than a blend. It would also rescale
   `nShell0` $\propto$ `params['Pb']` (`shell_structure.py:125-126`) and propagate through the whole
   shell solve. B11.B did not measure that scenario, but it measured its close analogue — replaying
   each driving row with the inner pressure set to $P_{\rm C3a}$ instead of `params['Pb']`, a
   $\times4.70$ (transition) / $\times6.17$ (momentum) change in `shell_n0`, comparable in size to
   $\alpha_p\approx6$ — and found the ionised layer thinning 79–83% and the dust-absorbed fraction
   moving $0.620\to0.455$ (transition) and $0.607\to0.395$ (momentum). It is not a one-line change.

**What is actually worth doing.** TRINITY already computes the quantity; it just never names it.
Emit, per snapshot, the **force ratio**

$$\frac{F}{\dot p} \;=\; \left(\frac{R_2}{R_1}\right)^{2}$$

through phases 1a, 1b and 1c — flagged invalid inside the `dt_switchon` window, where the ODE's
effective pressure is not $\dot p/(4\pi R_1^2)$. This is simultaneously TRINITY's exact force ratio
and, under the pressure match, Lancaster's exact $4\pi\mathcal{R}_w^2P_{\rm hot}/\dot p$, and it is
what Paper II's tabulated $\alpha_p$ approximates (§2.2). **Do not** emit the exact
`eq:alphap_shock` inversion $(R_2/R_1)^2+\tfrac3{16}(R_1/R_2)^2$ as the headline number: it is
correct as algebra but reports $1.1875$ where the force ratio is exactly 1, which is precisely the
regime the 1c→2 handover approaches. In phase 2, report $1$ by construction and say why.

That gives you a real, publishable comparison against Paper II `tab:cem_comp`. **The question it
answers is not "is $\alpha_p$ missing" but "does TRINITY's $\alpha_p$ collapse from
$(R_2/R_1)^2\gg1$ to 1 too abruptly at the 1c→2 handover?"** — a handover governed by
`ENERGY_FLOOR = 1e3` in `run_transition_phase.py:97`, not by any physical $\alpha_p$ criterion. If
$(R_2/R_1)^2$ is still $\sim$5 when $E_b$ crosses the floor, that is a genuine discontinuity in a
measurable quantity and a much better paper result than a fitted knob. If it has already fallen to
$\sim$1, the momentum phase is behaving exactly as Lancaster's theory says it should, and the
momentum-phase question is entirely about $P_{\rm HII}$ (§4.1).

Note also: `phaseSwitch_LlossLgain = 0.05` ⟹ $\theta = L_{\rm loss}/L_{\rm gain} = 0.95$ is the
`cooling_balance` trigger that ends phase 1a (`run_energy_phase.py:293` reads the threshold,
`:295` applies it) and drives the implicit→transition switch (`run_energy_implicit_phase.py:1252`
reads, `:1298` applies). It is **not** the energy→momentum handover, which is
`if Eb < ENERGY_FLOOR` at `run_transition_phase.py:768`. Revision 1 built an $\alpha_p$ estimator on `eq:tofap` at $\theta=0.95$; that route is now
demoted to a curiosity, and if you use it at all, carry Paper I's own conditions — it *"assume\[s\]
the modified momentum-driven solution"*, ignores turbulent motions in the WBB shell, and requires
$\alpha_p \gtrsim 1.25$.

### 4.5 $T_i$, $c_i$ and $\alpha_B$ are metallicity-blind

`TShell_ion = 1e4` and `caseB_alpha = 2.59e-13` are `run_const=True` and are not derived from
`ZCloud` (`registry.py:396,400` — no validator, no resolver; nothing in the tree derives either from
metallicity). But:

- **Geen 2019** compute $c_i$ from Cloudy as a function of $T_*$, $Z$, $n_i$ and $\mathcal{U}$
  (`fig:ionisedsoundspeed`), at fixed fiducial $n_i=50\,$cm$^{-3}$ and $\mathcal{U}=-2$. Their text
  quotes *"on the order of 10 km/s"* — for gas at $10^4$ K, not explicitly tied to solar $Z$; the
  higher low-$Z$ value ($\sim$15 km/s) is readable from that figure and a commented-out table, not
  from quotable body text — cite it as a figure read. Their coefficients scale as $C_w\propto c_i^{-3}$ (`wind:coefficient`) and
  $C_B\propto c_i^{-4}$ (`breakout:condition`).
- **Geen 2022** carry star-dependent $T_i$ (`table:app_starprops`) and
  $\sigma_d = 10^{-21}\,\mathrm{cm}^{-2}\,Z/Z_\odot$.
- **Lancaster's** $\zeta\propto c_i^{-1}$ (MD), $c_i^{-3/2}$ (ED).
- **Haid** measure $\bar T_{\rm HII}$ = 7160–8150 K and use $\alpha_B\propto T^{-0.83}$.

A factor 1.5 in $c_i$ is a factor $\sim$3 in $C_w$ and $\sim$5 in $C_B$. Since `ZCloud` already
switches SPS tracks and cooling tables, letting it also set $T_i$ (a small table, or the Geen 2019
fit) closes a real inconsistency — and moves the answer in the direction all three papers agree on:
**low $Z$ makes photoionisation relatively stronger.** Probably your cheapest genuinely-new physics.

### 4.6 Make the overflow transition an explicit event

$Q_{i,\rm abs}=Q_i f_{\rm abs,ion}$, so $P_{\rm C3a}\propto\sqrt{f_{\rm abs}}$ decays to zero as the
shell becomes transparent, and the `n_IF_Str > 0` gate fails outright at $f_{\rm abs}=0$. Meanwhile
$P_{\rm ext}$ is gated on `shell_fAbsorbedIon < 1.0` (`energy_phase_ODEs.py:235-238`, and identically
in all four runners). So the two respond to the same physical transition at different points, and
over the sequence photoionisation goes from accelerating to decelerating.

**A correction to revision 1, which had the $P_{\rm ext}$ gate backwards.** It said the condition is
"essentially always true, so $P_{\rm ext}$ is effectively always on". The opposite holds on the
branch this section is about: `shell_structure.py:230` returns `f_esc_ion = max(0.0, ...)`, so
$f_{\rm abs}$ is *exactly* 1.0 whenever the shell traps everything, and B11.0 measures exactly that
on 100% of transition and 76.5% of momentum driving rows (all of them at $L_w\times0.1$). There
$P_{\rm ext}$ is exactly **zero** — the inward photoionised term is off precisely where the
outward one dominates. (The ISM term is added separately and unconditionally beyond
$r_{\rm cloud}$, so $P_{\rm ext}$ is not identically zero out there.) $f_{\rm abs}<1$ is instead
common in the *energy* phase — `PLAN.md` records `frac_fabs_ge_099` falling to 0.000 on the weak-wind
configs — so the "always on" reading is right only where §4.6 does not apply.

Geen 2022 are careful here and should be quoted accurately: *"Reaching the overflow radius does not
guarantee an immediate and strong champagne flow"*; they describe a rarefaction wave moving back from
$r_i$ toward $r_w$ that eventually disperses the shell, and state *"the precise behaviour of the
\HII region after it reaches the overflow radius is mostly beyond the scope of this paper."* They do
**not** claim 1D models break there — they decline to model it. (Revision 1 said TRINITY's
degradation happens "precisely where Geen 2022 say a champagne flow begins", which contradicts the
quote it then gave.)

My suggestion is not to model the champagne flow either, but to **make the regime change explicit**:
an event in `phase_events.py` on $f_{\rm esc,ion}$ crossing a threshold, logged as a distinct fate /
`SimulationEndCode`, so a run coasting through overflow is visible in the output instead of looking
like ordinary deceleration. Geen 2022's `eqn:overflowcondition` gives an analytic overflow radius for
$\rho\propto r^{-\omega}$ as an independent cross-check — and §2.3's reference-mass difference makes
that comparison worth doing rather than assuming agreement.

Related: $P_{\rm ext}$ is evaluated from the **unshocked** profile at $r_{\rm shell}$
(`get_press_ion(rShell, ...)`) but applied over $4\pi R_2^2$. Fine for a thin shell; see §4.7.

### 4.7 State the thin-shell validity limit explicitly

Your B11.D wording is right and I would put it in the method paper verbatim in substance: on B3M's
momentum rows $dR_{\rm full}/R_2$ = 0.6723–1.3078 and $dR_{\rm ion}/R_2$ = 0.6579–1.3076, the shell
is 99.54% ionised, and both the ODE's thin-shell premise and C3a's sharp cavity/shell split are
outside their validity range. B12 measured it **worse** at low wind ($dR_{\rm ion}/R_2$ =
1.171–1.438). The trajectory crosses $dR/R_2\gtrsim1/3$ inside the *transition* phase.

This is not a defect with a proposed fix, and saying so plainly is stronger than not saying it. It
also bounds §4.6: the $P_{\rm ext}$ area mismatch is a real error once $dR/R_2\sim1$.

### 4.8 Diagnostics (cheap, and they would strengthen a paper)

1. **Emit $(R_2/R_1)^2$** as in §4.4, flagged invalid inside the `dt_switchon` window and reported
   as exactly 1 in phase 2, where the code asserts $\mathcal{R}_f=\mathcal{R}_w$.
2. **Emit $\zeta=R_{\rm eq}/R_{\rm St}$ and $R_2/R_{\rm ch}$**, stating the $\alpha_p$ used (both
   scale with it: $\zeta\propto\alpha_p^{1/2}$, $R_{\rm ch}\propto\alpha_p^2$). These place every
   TRINITY run directly on Paper I `fig:feedback_ratio` / Paper II `fig:feedback_ratio_comp`. Given
   that C3c's momentum-phase switch *is* $R_2=R_{\rm ch}$, `R2/Rch` is arguably the most informative
   single diagnostic your code could emit.
3. **`F_ram` is mislabelled in three places.** `energy_phase_ODEs.py:415`,
   `run_energy_implicit_phase.py:539` and `run_transition_phase.py:338` all set
   `F_ram = Pb * 4πR2²` — the *bubble* force, not the ram force — while `P_ram` is reported
   separately (and is 0 in the energy phase). `run_momentum_phase.py:272` is correct
   (`F_ram = P_ram * 4πR2²`). Reporting only, but it is what every force-budget figure consumes.
4. **The `n_IF_Str > 0` gate is vestigial.** Since C3c recomputes the density from $R_2$,
   `n_IF_Str` and its `min(n_IF_Str, shell_n0)` cap no longer feed $P_{\rm HII}$ — they only gate it.
   Retire the gate or replace it with the condition it intends ($Q_{i,\rm abs}>0$, $R_2>0$), which
   `get_phii_c3c` already checks internally. (Same finding as your B11.E.)
5. **Emit Geen 2019's $C_w$** alongside $\zeta$ — two independent wind/photoionisation orderings from
   quantities you already carry.

---

## 5. On the thin-shell / 1D approximation specifically

You flagged this as a constraint. Three observations, one of which reverses revision 1.

**It is not the limitation you might think for $P_{\rm HII}$.** Everything in §4.1–§4.3 is achievable
in 1D — Lancaster's entire CEM *is* a 1D model. Separating $\mathcal{R}_w$ from $\mathcal{R}_i$ costs
one algebraic root-find, not a dimension. Geen 2019/2022 likewise close the whole wind + PIR system
in 1D.

**It is the limitation for *half* of $\alpha_p$.** In 1D $A_w\equiv4\pi\mathcal{R}_w^2$, so the
fractal-area half of Lancaster's `eq:alphap_derive` is structurally unavailable. The dissipative half
is not: TRINITY carries it as the bubble energy budget, which is why $(R_2/R_1)^2\gg1$ in the energy
phase. So the correct statement is that TRINITY *can* have $\alpha_p>1$ and does, and what it cannot
do is generate the geometric enhancement — for which any hand-set value would be a calibration. Your
`cooling_boost_fA` / `cooling_boost_kappa` knobs are the legitimate hooks: $f_A$ *is* the $A_w$
factor, on the energy side. So if you ever do calibrate the geometry, calibrate it once, through
$f_A$, and let $\alpha_p$ follow — do not set the two independently.

**The thin-shell geometry biases you toward "confined" — but revision 1's proposed test had the wrong
sign.** Putting *all* swept mass in a thin shell maximises the shell density and hence radiation
trapping, relative to a turbulent, clumpy medium where photons leak through low-density channels.
The porosity form of this criticism is **Paper II's** (`subsec:role_of_turbulence`,
`app:supp_analysis`), which also quantifies the clumping of the background medium
($\mathfrak{C}=2.39$–10.3, `subapp:clumping`, measured after the turbulent evolution but before
feedback). Paper I `app:trapping` makes a related but distinct argument — from background *density
gradients* and cluster-vs-single-star geometry — and itself assumes a constant-density shell.

Revision 1 suggested a clumping factor $\mathfrak{C}$ multiplying $\alpha_Bn^2$. **That moves the
wrong way:** raising the effective recombination rate *shrinks* $R_{\rm St}$, *lowers*
$P_{\rm C3a}\propto R_{\rm St}^{3/2}$, and makes the confined branch fire *more*, not less. The
correct 1D proxy for photon leakage is a **covering/absorbed-fraction** knob — an $f_{\rm ion}<1$
applied to $Q_i$ before the shell solve, representing the fraction of solid angle over which photons
escape through channels. That is also closer to what Paper I `app:trapping` is actually complaining
about, and it composes with Geen 2022's $\Omega<4\pi$ geometry (§2.3).

---

## 6. Verification

Revision 2 went through two adversarial passes after drafting — one against the papers, one against
the code and workstream data. §6.2 records what revision 1 got wrong; §6.3 records what those two
passes then found in revision 2's own first draft.

### 6.1 What was checked, and how

| Claim | Result |
|---|---|
| `solve_R1` root $\Rightarrow P_b=(\gamma-1)\tfrac{3}{4\pi}L/(vR_1^2)$; $=\dot p/(4\pi R_1^2)$ at $\gamma=5/3$ | exact identity (SymPy `simplify == 0`) |
| `pRam` with $v_{\rm mech}=2L/\dot p_{\rm tot}$ $\Rightarrow F_{\rm ram}=\dot p_{\rm tot}$ | exact |
| C3c crossover (momentum phase) $=R_{\rm eq}^4/R_{\rm St}^3=R_{\rm ch}$ | ratio $=1$ exactly |
| $\alpha_p$ from Weaver-convention $R_2/R_1$ via `eq:alphap_shock` | $=(R_2/R_1)^2+\tfrac{3}{16}(R_1/R_2)^2$ exactly (SymPy); $+1.16\%$ at $R_2/R_1{=}2$, $+0.23\%$ at 3 |
| $\alpha_p=1\Leftrightarrow R_2/R_1=\sqrt3/2$ in TRINITY's convention | $0.8660254$ |
| C3c crossover vs $f_{\rm abs}$ | $R_2^{\rm cross}\propto f_{\rm abs}^{-1}$ (SymPy solve), i.e. **outward** |
| $F_{\rm CEM}\to\alpha_p\dot p$ as $\mathcal{R}_w/R_{\rm ch}\to0$ | 1.00067 at $10^{-3}$ |
| $F_{\rm CEM}\to F_{b,\rm Sp}$ as $\mathcal{R}_w/R_{\rm ch}\to\infty$ | 1.000067 at $10^{4}$ |
| $F_{\rm sum}/F_{\rm CEM}$ at $\mathcal{R}_i=R_{\rm ch}$ | **1.342** (Lancaster state $\sim$1.35) |
| $F_{\max}/F_{\rm CEM}$ at $\mathcal{R}_i=R_{\rm ch}$ | 0.671 |
| `eq:tofap` inverted for $\alpha_p$ | round-trips $\theta$ to $10^{-6}$ |
| $\zeta(\alpha_p=6.25)$, Paper II cluster | 0.991 (paper quotes $\approx0.98$) |
| Lancaster's exact force $\tfrac34x^2$ under the pressure match | $=(R_2/R_1)^2$ exactly |
| C3c crossover $=R_{\rm ch}$ requires $\alpha_p=1$ | crossover $=R_{\rm ch}/\alpha_p^2$ in general |
| $F_{\rm sum}/F_{\rm CEM}$ true maximum | 1.344 at $\mathcal{R}_i/R_{\rm ch}=0.79$, not at 1.00 |

Code claims re-checked against the staged source this revision: `run_momentum_phase.py:587-588,893`
(`R1 = R2`), `:272` (`F_ram = P_ram·4πR2²`), `:585,669,891` (`Pb = pRam`);
`energy_phase_ODEs.py:415`, `run_energy_implicit_phase.py:539`, `run_transition_phase.py:338`
(`F_ram = Pb·4πR2²`); `get_bubbleParams.py:365` (the `params['Pb']` comparison), `:495-503`
(`dt_switchon` ramp); `shell_structure.py:119-120,125-126,158,182,243,253`;
`run_energy_phase.py:55` (`TFINAL_ENERGY_PHASE = 3e-3`), `:293`;
`run_energy_implicit_phase.py:1252`; `run_transition_phase.py:97` (`ENERGY_FLOOR = 1e3`);
`registry.py:396,400,407`. Every LaTeX label was re-checked against the cited paper, including which
of `eq:Rch_def` / `eq:Rch_Req_rel` / `eq:Rch_app` carries which statement.

### 6.2 What revision 1 got wrong

Recorded rather than quietly patched, because the point of the exercise was that each claim be true.

| # | Revision 1 claim | Status |
|---|---|---|
| 1 | "$\alpha_p\equiv1$ is almost certainly why your momentum phase is HII-dominated"; add an `alpha_p` knob | **WRONG.** Phase 2 applies $F=\dot p$ identically via `pRam`; Batch 12 bounds the required factor at $\gtrsim14$. §4.4 replaces it |
| 2 | "Do **not** report $(R_2/R_1)^2$; it converges to $\tfrac43\times$ Lancaster's $\alpha_p$" | **WRONG, backwards.** With Weaver's convention the $4/3$ cancels: $\alpha_p=(R_2/R_1)^2+\tfrac3{16}(R_1/R_2)^2$ |
| 3 | Haid's non-additivity "goes the opposite way to Lancaster's analytic result" | **WRONG.** `eq:force_low_eta` gives super-additivity at low $\zeta$; Paper II puts Haid at $\zeta\le0.35$ — same branch |
| 4 | "Geen & de Koter 2022 — you already implement their model" | **WRONG chronology.** Geen 2022 cites Rahner 2017. Common Draine (2011) inheritance |
| 5 | "You are strictly ahead" of Geen 2022 | **WRONG.** They carry $Z$-dependent $\sigma_d$ and star-dependent $T_i$ — exactly §4.5's charge against TRINITY |
| 6 | §4.4 shell-mass adjustment is "currently near-irrelevant (thin skin)" | **WRONG, measured false.** $dR_{\rm ion}/dR_{\rm full}$ median 0.9954; adjustment worth $+8.55\%/+9.22\%$ in $R_2$ |
| 7 | The $-33\%$ / $+34\%$ table labelled "1a/1b" and "momentum" | **WRONG scope.** The closure is MD-phase only; the ED case needs Paper I `subsec:ed_jfb`'s energy equation |
| 8 | Partial absorption shifts the crossover by $f_{\rm abs}^{1/3}$ | **WRONG** in exponent and sign: $f_{\rm abs}^{-1}$, outward |
| 9 | "Paper I levels three criticisms at WARPFIELD" | **WRONG.** Eight; the omitted ones include its central thesis |
| 10 | "`frac HII-dominated = 0.0000` on every energy and implicit row of all five configs" | **STALE.** Batch 7 records `B3MW001` at 78.4% HII-dominated in the energy phase |
| 11 | "`F_ram` is mislabelled in two phases" | **WRONG count.** Three: energy, implicit, transition |
| 12 | Seam ratios "0.995–0.999" quoted for the handover generally | **UNDER-QUALIFIED.** That is the transition→momentum seam; others are 0.89–0.92, 0.53–0.96, 0.86–0.99 |
| 13 | Identity I stated without qualification | **INCOMPLETE.** Holds for stored `Pb` always; for the ODE-effective pressure only outside the `dt_switchon` window (up to the first third of 1a; up to $3.2\times$ difference inside) |
| 14 | §5's clumping factor $\mathfrak{C}$ on $\alpha_Bn^2$ | **WRONG SIGN.** It makes the confined branch fire *more*. Use $f_{\rm ion}<1$ on $Q_i$ instead |
| 15 | "$\theta=0.95$ is your transition trigger" used as an energy→momentum handover value | **MISLABELLED.** It ends 1a and drives 1b→1c; the energy→momentum handover is `Eb < ENERGY_FLOOR = 1e3` |
| 16 | "the CEM ignores … on-the-fly bubble cooling … TRINITY has all seven" | **OVERSTATED.** The ED-CEM does solve an energy equation; six, and what it lacks is a dissipation model |
| 17 | Paper II "independently confirms Haid's direction" | **OVERSTATED.** Different quantity ($\alpha_p$, not $p_{\rm Combi}$ vs the sum) |
| 18 | Geen's overflow criterion identified with TRINITY's | **IMPRECISE.** TRINITY matches Geen's *analytic* form ($M(<r_w)$), not the numerical one ($M(<r_i)$); it overflows earlier by $(r_i/r_w)^{3-\omega}$ |
| 19 | "$P_{\rm C3a}/P_{\rm ram}=3.8$–7.6" | **SUPERSEDED** by measurement: 5.091–7.156, median 6.165 (B3M momentum); 13.667–14.369 at $L_w\times0.1$ |
| 20 | Paper I's CEM pressure balance described as a result | **IMPRECISE.** Imposed: *"we now imagine"*, *"forced to be at the same pressure"* |
| 21 | Table 1 cited `energy_phase_ODEs.py:385` as the 1c site | **HALF RIGHT.** `:385` really is unreachable, but `run_transition_phase.py:331` is *reporting only* — the live 1c site is `energy_phase_ODEs.py:253` (see §1). Revision 2 corrected this in the wrong direction on its first pass and re-corrected it |
| 22 | "three different thresholds" in §4.6 | **TWO** — but see §6.3: the `P_ext` half was itself wrong |

### 6.3 What the *second* verification pass changed

Revision 2 was itself run past two independent adversarial passes — one against the papers, one
against the code and the workstream data — before release. Nine more findings, all incorporated
above; the four that changed a conclusion:

| Draft-2 claim | Status |
|---|---|
| "$P_{\rm ext}$ is effectively always on" | **REFUTED.** `f_abs` saturates at exactly 1.0 on 100% of transition and 76.5% of momentum driving rows, so $P_{\rm ext}$ is exactly **zero** there. The gate is on in the *energy* phase instead (§4.6) |
| "$\alpha_p\equiv1$ is forced by `R1 = R2`" | **REFUTED as causal.** `params['R1']` is read in exactly one place (`get_bubbleParams.py:115`, energy/implicit code); in phase 2 it is bookkeeping. What forces $F=\dot p$ is `pRam` with $v_{\rm mech}=2L/\dot p$ (§1, Identity II) |
| "a 1D code cannot produce $\alpha_p>1$ by this mechanism at all" | **REFUTED.** `eq:alphap_derive` has *two* factors; only the geometric one is 1 in 1D. TRINITY has $\alpha_p=(R_2/R_1)^2\gg1$ in its own energy phase (§2.2, §4.4, §5) |
| "`run_transition_phase.py:331` is the live 1c site" | **REFUTED.** The transition RHS delegates to `get_ODE_Edot_pure`; the live sites are `energy_phase_ODEs.py:253,256` and `run_momentum_phase.py:445` (§1) |

Five smaller ones: the $R_{\rm eq}$ definition is `eq:RE_MD_def`, not `eq:eta_def`; the $\zeta=0.35$
statement is in `subsubsec:ideal_sim_review`, not `app:sim_comp`; `eq:force_low_eta` has no $c_i^2$;
the ramp discrepancy is $3.31\times$, not $3.2\times$; and Paper I's eight "distinct" criticisms are
better grouped as six, with two I had missed (constant-density PIR, and shell inertia).

Two numbers I could not source and therefore removed: the lower bound `0.487` on Batch 7's
confinement ratio (it appears only in revision 1 of this document, and
`data/b7_confinement_screen.csv` is not in my staged copy).

### 6.4 What I did *not* verify

- **Anything requiring a TRINITY run.** Every measured number in this revision is taken from your
  committed `data/*.csv` and the Batch 11/12 write-ups, cited as such. I re-derived none of them.
- **Whether $(R_2/R_1)^2$ actually reaches Paper II's 4.6–6.8 before the 1c→2 handover.** That is the
  central open question of §4.4 and it is unmeasured. The cheapest test is an offline pass over
  existing snapshots — no solver run.
- **Geen 2019's low-$Z$ $c_i\approx15\,$km/s.** Readable from `fig:ionisedsoundspeed` and a
  commented-out table; the body text quotes only *"on the order of 10 km/s"*.
- **Paper II's $\alpha_p$ at higher resolution.** The MWR sequence 5.57 → 6.20 → 6.82 is not
  converged and the paper says so.

**Staleness caveat:** `docs/dev/` carries its own "may be out of date" banner, and its
`LITERATURE_ASSESSMENT.md` is revision 1 of *this* document, under a maintainer note that a
correction was inbound. This is that correction. Its §4 items should now be re-read against §6.2
above before any of them is acted on. Every code claim here was re-checked against source at the
2026-08-18 pull; `trinity/` was byte-identical to the copy read for revision 1.

---

## 7. Suggested order of work

1. **Emit the diagnostics** (§4.8): $\alpha_p=(R_2/R_1)^2+\tfrac3{16}(R_1/R_2)^2$ through 1a/1b/1c,
   $\zeta$, $R_2/R_{\rm ch}$, Geen's $C_w$. Zero risk, no solver run, and it answers "where do my
   runs sit on `fig:feedback_ratio`".
2. **Offline: does $(R_2/R_1)^2$ reach 4.6–6.8 before the 1c→2 handover?** (§4.4) One pass over
   existing snapshots. This is the question revision 1 should have asked.
3. **Run B11.G** — score the shipped closure against Geen 2019 `wind:photoequilibrium` +
   `wind:windpressurebalance` / Geen 2022 `eqn:photoionisation_equilibrium_uniform` +
   `eqn:PwPibalance` on your own trajectory. Cheap, offline, and it is the direct input to §4.1.
4. **The balance volume** (§4.1) — K5 as the minimal move, K6 as the right one. Pre-register the
   ablation as you did for C3c; it will move fates.
5. **Shell-mass adjustment** (§4.2) — already measured at $+8.6$–$9.2\%$; land it with Lancaster's
   own consistency caveat stated.
6. **$Z$-dependent $T_i$, $c_i$, $\alpha_B$** (§4.5).
7. **Paper-scale:** three-radius model, the coupled closure across all four phases, an $f_{\rm ion}$
   leakage knob (§5), and the stated thin-shell validity limit (§4.7) in the methods section.

---

## Sources

**Papers (project docs)**

- `lancaster2025.tex` — Lancaster, J.-G. Kim, Bryan, Menon, Ostriker & C.-G. Kim, *The Co-Evolution
  of Stellar Wind-blown Bubbles and Photoionized Gas I*
- `lancaster2025b.tex` — *…II: 3D RMHD Simulations and Tests of Semi-Analytic Models*
- `geen2019.tex` — Geen, Pellegrini, Bieri & Klessen, *When H\,II Regions are Complicated:
  Considering Perturbations from Winds, Radiation Pressure, and Other Effects*
- `geen2022.tex` — Geen & de Koter, *Bottling the Champagne: Dynamics and Radiation Trapping of
  Wind-Driven Bubbles around Massive Stars*
- `Haid118_arxiv.tex` — Haid, Walch, Seifried, Wünsch, Dinnbier & Naab, *The relative impact of
  photoionizing radiation and stellar winds on different environments*

**TRINITY source (read-only)**

- `trinity/bubble_structure/get_bubbleParams.py`, `bubble_luminosity.py`
- `trinity/phase1_energy/energy_phase_ODEs.py`, `run_energy_phase.py`
- `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py`
- `trinity/phase1c_transition/run_transition_phase.py`
- `trinity/phase2_momentum/run_momentum_phase.py`
- `trinity/shell_structure/shell_structure.py`, `get_shellODE.py`
- `trinity/sps/update_feedback.py`, `trinity/_input/registry.py`

**Workstream docs**

- `docs/dev/phii-identity/PLAN.md` — §0 (C-0 carve-out), §3c.1, §6b, Batch 7, Batch 11 (B11.0,
  B11.A–D), Batch 12, §7.1 candidate register (K1–K9)
- `docs/dev/phii-identity/README.md` — §5, §6
- `docs/dev/phii-identity/LITERATURE_ASSESSMENT.md` — revision 1 of this document, with the B11
  cross-check
- `data/b7_confinement_screen.csv`, `b10_wind_profile.csv`, `b11_mass_ledger.csv`,
  `b11_photon_ledger.csv`, `b11_mass_dynamics.csv`, `b12_lowwind_mass_ledger.csv`,
  `b12_lowwind_mass_dynamics.csv`
