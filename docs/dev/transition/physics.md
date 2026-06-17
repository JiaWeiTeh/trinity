# Physics of the transition-trigger candidates F0–F4

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
> a committed artifact (a CSV/table under `docs/dev/data/`, or a force-added
> harness/figure in the relevant `docs/dev/<workstream>/` folder as the hybr work did) — never left in `/tmp` or
> an untracked `outputs/`. A future visit must be able to reproduce or compare
> against the numbers **without re-running**; record the exact config + command
> that produced each artifact.

**About this document**
- **Status (verified 2026-06-17):** 🔵 **REFERENCE** — the physics/equations behind the candidate-trigger *families* named in `TRIGGER_PLAN.md` (the menu) and harvested in `P0.md` (the results). Equation forms cross-checked against `make_p0_figures.py` and the harvest columns; constants re-verified against current source on the date shown. Re-verify per banner.
- **Type:** reference — derivations and code-mapping for F0–F4, kept distinct from the *plan* (what to test) and the *results* (what fired where).
- **Workstream:** `transition/` — the implicit→momentum transition trigger.
- **Where it sits:** `TRIGGER_PLAN.md` (plan / family menu) → **this (the equations behind the menu)** → `P0.md` (firing epochs) → `pshadow-design.md` (chosen-criterion design).
- **Code it concerns:** the quantities harvested in `docs/dev/transition/harness/harvest.py` and re-derived for the figures in `make_p0_figures.py`; the live F0/F4 predicates in `trinity/phase_general/transition_shadow.py`; the bubble pressure relation `bubble_E2P` (`trinity/bubble_structure/get_bubbleParams.py`).
- **Linked files & data:** `TRIGGER_PLAN.md`, `P0.md`, `pshadow-design.md`; figures `docs/dev/transition/figures/p0_*.png`; data `docs/dev/data/transition_*.csv`.

---

## Setup: the quantities the criteria are built from

The implicit phase (1b) integrates the hot bubble's energy `Eb` while the shell
expands. Every candidate is a test, evaluated each segment, on a small set of
harvested quantities (column names from the harvest CSVs in parentheses):

| symbol | meaning | code / column |
|---|---|---|
| $L_\mathrm{gain}$ | **instantaneous** mechanical luminosity injected by the cluster | `Lgain` $= L_\mathrm{mech}$ (`feedback_post.Lmech_total`) |
| $L_\mathrm{loss}$ | **radiative** cooling of the bubble interior $+$ the leak term | `Lloss` $= $ `bubble_LTotal` $+$ `bubble_Leak` |
| $E_b$ | thermal energy stored in the hot bubble | `Eb` |
| $R_2$ | outer shell radius; $R_1$ inner (wind/contact) radius | `R2`, `R1` |
| $v_2$ | shell velocity $\dot R_2$ | `v2` |
| $R_\mathrm{cloud}$ | cloud (GMC) radius | `rCloud` |
| $P_b$ | bubble thermal pressure | `bubble_E2P(Eb, R2, R1, γ)` |

Two facts fix the normalisation and are worth stating once:

- **$L_\mathrm{loss}$ is pure radiative cooling — no $P\,dV$.** `bubble_LTotal`
  $= L_\mathrm{bubble} + L_\mathrm{conduction} + L_\mathrm{intermediate}$, each a
  radiative integral $\int \chi_e\, n^2\, \Lambda(T)\, 4\pi r^2\,dr$ (CIE) /
  $\int \dot u(n,T,\phi)\, 4\pi r^2\,dr$ (non-CIE) over the interior. The $P\,dV$
  work $4\pi R_2^2\, v_2\, P_b$ is carried *separately* by the βδ energy balance,
  so the ratios below are a clean cooling-vs-injection fraction (no
  double-counting). See `bubble_luminosity.py`.
- **Bubble pressure** from energy (`get_bubbleParams.py:228`), $\gamma = 5/3$:

$$P_b \;=\; (\gamma-1)\,\frac{E_b}{\tfrac{4\pi}{3}\,(R_2^3 - R_1^3)} \;\xrightarrow[\;R_1\ll R_2\;]{}\; \frac{E_b}{2\pi R_2^3}.$$

All firing epochs use the **left-rectangle rule** (snapshot $k$ = state *before*
segment $k$); cumulative integrals below respect it.

---

## F0 — instantaneous rate-ratio (the current/baseline trigger)

The live terminator of the implicit phase, the WARPFIELD-style energy-retention
test (Rahner et al. 2017/2019). With threshold $\varepsilon$ = `phaseSwitch_LlossLgain` (default $0.05$):

$$\boxed{\;\frac{L_\mathrm{gain} - L_\mathrm{loss}}{L_\mathrm{gain}} \;<\; \varepsilon\;}
\qquad\Longleftrightarrow\qquad L_\mathrm{loss} \;\ge\; (1-\varepsilon)\,L_\mathrm{gain}.$$

It fires when cooling radiates $\ge 95\%$ of the **instantaneous** injected power
— i.e. *"$E_b$ stops growing,"* not *"the thermal drive stops pushing the
shell."*

**Known pathology — the reset.** Every new feedback episode (WR onset, first SN)
spikes the *denominator* $L_\mathrm{gain}=L_\mathrm{mech}$, so the ratio jumps
**up and away** from $\varepsilon$ (harvested $0.44 \to 0.67$ across the WR surge
— *more* energy-driven) exactly when a new source switches on. This is an
artifact of testing the instantaneous numerator, not physics; integrating over
episodes (F1) removes it. F0/F4 are the two predicates implemented live in
`transition_shadow.py`. F1–F4 below are the alternatives the plan weighs against it.

---

## F1 — cumulative energy retention

*(Mac Low & McCray 1988; Nath et al. $\eta\!\sim\!0.25$; Sharma et al. $0.2$–$0.4$; WARPFIELD calibration — verify each)*

Integrate injection and cooling over the bubble's history:

$$E_\mathrm{gain}(t) = \int_0^t L_\mathrm{gain}\,dt', \qquad
  E_\mathrm{loss}(t) = \int_0^t L_\mathrm{loss}\,dt', \qquad
  f_\mathrm{cum}(t) \equiv \frac{E_\mathrm{loss}(t)}{E_\mathrm{gain}(t)}\;\;(\texttt{frac\_cum}).$$

The **retained fraction** is $\eta_\mathrm{ret}(t) = 1 - f_\mathrm{cum} \simeq
E_b / E_\mathrm{gain}$ — the share of all injected energy still resident in the
bubble. The trigger fires when retention falls below $\eta$ (with $\eta \approx
0.2$–$0.4$):

$$\boxed{\;f_\mathrm{cum} \;>\; 1-\eta \;}
\qquad\Longleftrightarrow\qquad \eta_\mathrm{ret} \;<\; \eta.$$

**Why it cures the F0 reset.** $E_\mathrm{gain}(t)$ is **monotone
non-decreasing**, so a transient spike in $L_\mathrm{gain}$ adds to the
denominator's *running total* but cannot push $f_\mathrm{cum}$ back below the
threshold — the criterion never resets across a WR/SN episode. This is the single
strongest argument for F1 over F0. It is anchored to the retained-energy fraction
of adiabatic bubble theory (Weaver et al. 1977 gives $E_b = \tfrac{5}{11}L_w t$
in the pure-adiabatic limit; $\eta$ is the literature retained-energy fraction).

**Caveat.** Keep three fractions distinct and do not conflate them: $0.27$ (our
internal adiabatic algebra, order-of-magnitude only), $0.35$ (CMW outer-shock
$27/77$, a *different* quantity), and $\eta \approx 0.25$ (the retained-energy
calibration anchor used here). Verify the $\eta$ range against each cited source.

---

## F2 — timescale ratio (cooling time vs dynamical time)

*(Mac Low & McCray 1988; Koo & McKee 1992)*

Compare how fast the interior radiates its thermal energy to how fast the bubble
expands:

$$t_\mathrm{cool} = \frac{E_b}{L_\mathrm{loss}}, \qquad
  t_\mathrm{dyn} = \frac{R_2}{v_2}, \qquad
  \boxed{\;\frac{t_\mathrm{cool}}{t_\mathrm{dyn}} \;<\; k\;}\quad (k \in \{1,2,3\})\;\;(\texttt{F2\_tcool\_tdyn}).$$

When $t_\mathrm{cool} < k\,t_\mathrm{dyn}$ the hot interior loses its thermal
energy faster than it can expand, so it can no longer sustain the adiabatic
over-pressure that drives the shell — the bubble becomes momentum-driven.

**Caveat (do not mis-attribute in the paper).** These **instantaneous** forms are
*our* construction. Mac Low & McCray's actual $t_\mathrm{cool}$ is a **cumulative
balance** $\int L_\mathrm{loss}\,dt = E_b(t_R)$, and their $t_\mathrm{dyn}$ is a
**scale-height crossing**, not $R/v$. Label the instantaneous version as ours.
(P0 found F2 fires far too early — $\sim 3$–$7$ kyr, two to three orders of
magnitude before the $E_b$-peak — so it was eliminated, but the form is recorded
here for completeness.)

---

## F3 — force / continuity (thermal drive vs surviving forces)

*(BETADELTA plan §5)*

A continuity-preserving test: the thermal-pressure force on the shell becomes
subdominant to the forces that *survive* into the momentum phase (ram + radiation
pressure). With the thermal drive

$$F_\mathrm{thermal} = 4\pi R_2^2\, P_b, \qquad P_b = (\gamma-1)\,\frac{E_b}{\tfrac{4\pi}{3}(R_2^3-R_1^3)},$$

the criterion is

$$\boxed{\;\frac{4\pi R_2^2\, P_b}{F_\mathrm{ram} + F_\mathrm{rad} + \dots} \;<\; \mathcal{O}(1)\;}.$$

It is *continuity-preserving* because it is the **same** force comparison the
transition/momentum runner already makes when it drives the shell on
$\max\!\big(P_b,\; P_\mathrm{HII} + P_\mathrm{ram}\big)$ — so switching when the
thermal term drops below the others injects no pressure discontinuity at the
hand-off. (P0 eliminated F3 alongside F2 for the harvested configs; retained here
because for non-cooling regimes a force test is the physically motivated
alternative to a cooling test.)

---

## F4 — blowout (geometric)

The shell escapes the cloud:

$$\boxed{\;R_2 \;>\; R_\mathrm{cloud}\;}\;\;(\texttt{R2\_over\_rCloud} > 1).$$

For a **steep $r^{-2}$ halo** the bubble expands into ever-lower density, so the
cooling rate $L_\mathrm{loss} \propto n^2$ collapses and **no cooling family
(F0–F2) ever fires** — the cumulative integral $\int L_\mathrm{loss}\,dt$ may
never accumulate enough to cross any threshold. The physical fate is then
**blowout**: once $R_2 > R_\mathrm{cloud}$ the shell breaks out of the GMC and the
thermal phase is over regardless of the energy budget. This is why F4 (and F3)
must stay in the candidate set — and why the eventual trigger is
**profile-dependent**, not a single scalar.

P0 confirmed this empirically: the steep $\alpha=-2$ (4 Myr) run **blows out at
$2.728$ Myr**, well before the WR surge ($3.178$ Myr) and with `ratio_F0` never
approaching $0.05$. F4 is now wired as a live implicit-phase terminator under
`transition_trigger='cooling_or_blowout'` (F0 ∨ F4, F0-first; see
`pshadow-design.md` and `transition_shadow.py`).

---

## How the families relate (and where they fire)

- **F0 vs F1** — same numerator/denominator quantities, but F1 integrates them.
  F1's only structural advantage is the no-reset property; P0 found F1 does **not**
  fire earlier than F0 on the flat configs (they coincide at the $E_b$-peak), so
  the extra machinery buys robustness, not a different epoch.
- **F2, F3** — timescale/force reformulations; both fire far from the $E_b$-peak
  on the harvested configs and were eliminated at gate G0.
- **F4** — the only *non-cooling* family, and the one that fires for the steep
  crux case where every cooling family is silent.

The firing epochs for all families across the five configs are tabulated and
plotted in `P0.md` and `figures/p0_divergence_map.{png,csv}`; the reset pathology
(F0 up, F1 down) is `figures/p0_reset_pathology.png`; the blowout crux is
`figures/p0_overlay_steep_long.png`. Regenerate every figure offline (no sim) with:

```
python docs/dev/transition/harness/make_p0_figures.py
```
