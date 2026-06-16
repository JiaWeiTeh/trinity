# `T_init = 3e4` sensitivity: is the bubble boundary anchor a relabel-only knob?

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
- **Status (verified 2026-06-16):** 🟡 **PARTIAL** (verified 2026-06-16) — the study concluded (3e4 is conservative); recommendation #3 (drop the linear L3 patch) is not implemented.
- **Type:** study — a thread-pinned sweep testing whether the `T_init = 3e4` bubble-boundary anchor is a relabel-only knob or actually shifts the bubble cooling luminosity `L_total`.
- **Workstream:** `misc/` — standalone (`T_init` / bubble-boundary anchor sensitivity).
- **Where it sits:** standalone — companion/follow-up to the integrator-robustness boundary-transient discussion; concerns phase0_init / the bubble boundary (these have no before/after within misc/).
- **Code it concerns:** the `_T_INIT_BOUNDARY` (3e4 K) anchor in bubble-structure luminosity and the linear "L3" patch over `[1e4, T_init]`; the full `get_bubbleproperties_pure` pipeline (R1/Pb → fsolve `dMdt` → ODE → luminosity).
- **Linked files & data:** code `trinity/bubble_structure/bubble_luminosity.py`; harness `tools/tinit_sweep/` (`run_sweep.py`, `sweep.py`, `README.md`), `tools/bubble_audit/audit.py`; related docs `docs/dev/bubble/integrator-robustness.md`.

Companion to `docs/dev/bubble/integrator-robustness.md` (the "T_init=3e4"
boundary-transient discussion). That doc established *what* the 3e4 boundary is
(a numerical anchor, not a physical temperature). This doc answers the
follow-up: **does the bubble cooling luminosity actually depend on the choice of
that anchor, and is the test of that claim reproducible rather than lucky?**

## TL;DR

- **The test harness is robust and deterministic** (the headline requirement):
  across 5 separate **thread-pinned** processes, every `L_total` is
  **bit-identical** (`float.hex`); the baseline re-solve reproduces the dumped
  converged `dMdt` to `rel ≤ 7e-8`; no run passes by FP coincidence.
- **The physics answer is asymmetric.** `L_total` is **robust to *lowering*** the
  anchor (≤0.44 % down to `T_init=1.5e4`) but **drifts *up* if raised**
  (+0.98 % at 4e4, **+2.71 % at 5e4**).
- **The mover is L3 — the linear "intermediate" patch over `[1e4, T_init]`.**
  `L_total`'s deviation tracks the L3 fraction nearly 1:1. The wider that
  interval (higher `T_init`), the more the *linear* patch distorts the cooling
  integral. This is exactly the "region-3 patch is inadequate" failure mode the
  physics review predicted.
- **Verdict.** `3e4` is a **defensible, conservative** numerical knob: at 3e4 the
  L3 patch is only 0.15–0.43 % of `L_total`, and the result is robust at-or-below
  it. It is **not** a free relabel *upward*. Full `T_init`-independence would
  require integrating the ODE down to the `1e4` floor (eliminating the linear
  patch) or a better-than-linear patch — a separate physics change.

## Method (why it cannot pass by coincidence)

`tools/tinit_sweep/` (see its README). For each captured bubble state, the
**full** production pipeline `get_bubbleproperties_pure` (R1/Pb → fsolve `dMdt`
→ initial conditions → ODE → luminosity) is re-run while varying only the single
coupled knob `bl._T_INIT_BOUNDARY` over `{1.5e4, 2e4, 3e4, 4e4, 5e4}` K.

That constant is the byte-identical extraction of the former three `3e4`
literals (anchor + fsolve rejection floor + penalty), so the sweep moves all
three together exactly as a real change would.

Five gates, K=5 repeats:

| Gate | Definition | Result |
|---|---|---|
| 1 Fidelity | baseline re-solve reproduces dumped `dMdt` (`rel<1e-3`) | **PASS** (≤7e-8) |
| 2 Determinism | `L_total` bit-identical across K **separate pinned** processes | **PASS** |
| 3 Contrast | does unpinned vary? (earns determinism) | unpinned *also* stable on this box |
| 4 Sensitivity | `|L(T_init)−L(3e4)|/L(3e4) ≤ 1%` over the grid | **FAIL at 5e4 only** |
| 5 Robustness | crashes counted, never silently a pass | **PASS** (0 crashes) |

States: 12 fresh dumps (current code) from the smoke param
(`mCloud 1e5 / sfe 0.3`), early energy phase, all at `(β+δ)=0.629`, spanning
`Eb ≈ 9e1 → 1.25e5` (≈3 dex). `tools/bubble_audit/audit.py` independently
confirms these states reproduce their dumped `T_array` **bit-exactly**
(`max_rel = 0`), so the operating point is faithful — unlike the committed
`outputs/mockOutput` jsonl, whose recorded `L` is ~12.5 % off current code
because it predates the mu-audit / units / grid-smoothing commits (#656–#658).

## Results

`L_total` relative to the `T_init=3e4` baseline, and the L3 fraction, per state
(state index ≈ evolution / increasing `Eb`):

```
state    L_total(3e4)   rel-to-baseline [1.5e4  2e4   3e4   4e4    5e4]    L3/L_total [1.5e4 .. 5e4]
0000     4.97e6         -0.44% -0.28%  0%  +0.44% +1.05%                   0.01% .. 1.58%
0005     1.57e7         -0.29% -0.27%  0%  +0.73% +1.94%                   0.00% .. 2.52%
0011     3.57e7         -0.39% -0.35%  0%  +0.98% +2.71%                   0.05% .. 3.00%
```

- Monotone in `T_init`; deviation grows with `Eb` (at 5e4: +1.05 % early →
  +2.71 % late).
- L3 fraction grows from ~0 % (1.5e4) to ~3 % (5e4); `L_total` deviation tracks
  it. L1/L2 move <1 % — the conduction/CIE regions are integrated by the ODE and
  are stable; only the *patched* region is sensitive.
- No boundary `T`-dip (`T_min == T_init`) appeared in any state at `(β+δ)=0.629`.

## Recommendation

1. **Keep `_T_INIT_BOUNDARY = 3e4` (or lower it).** It is conservative; the
   result is robust at-or-below it. If anything, *lowering* toward ~2e4 shrinks
   the L3 patch and the `T_init`-sensitivity (no crashes were observed down to
   1.5e4 — but stiffness/crash risk near the `T→0` edge should be re-checked on
   production-scale clouds before adopting a lower default).
2. **Do not raise it.** ≥4e4 pushes `L_total` past the 1 % band via the crude
   L3 patch.
3. **The real fix for strict `T_init`-independence** is to remove the linear L3
   patch — integrate the conduction ODE down to the `1e4` `_coolingswitch`
   floor, or replace the linear `interp1d` over `[1e4, T_init]` with the
   conduction similarity profile. Separate, physics-accuracy change; validate
   against this harness.

## Limitations

- **One `(β+δ)=0.629` regime.** The smoke states are early energy phase. The
  `(β+δ)→negative` transition regime (where `bubble-integrator-robustness.md`
  §I.5 located the boundary dip) needs later-phase states (a longer run) — not
  covered here. The dip did not appear at `(β+δ)=0.629`.
- **One param class** (`mCloud 1e5 / sfe 0.3`). The trend (L3-driven, asymmetric)
  is mechanistic and expected to generalize, but the *magnitude* of the upward
  drift may differ for other clouds.
