# Patch: five α_p columns into `paper/II-survey/reduce_survey.py`

**Not a new script.** `reduce_run()` already streams every snapshot of every run exactly once
and accumulates side tables from that single pass. The α_p question needs five scalars per
run, so it is three inserts into that loop and zero extra I/O — no second pass over
`gridsweep_v2/`, no new file to pull, no `--limit` and no run selection, because five floats
× 63,360 runs is noise inside a `summary.csv` that already carries dozens of columns and is
pulled unconditionally by `sync.sh down`.

Line numbers are against `reduce_survey.py` @ `mtimeMs 1787096047609`. Anchors are quoted so
the patch survives drift.

---

## 1. Accumulators — after `budget_series = []` (~line 335)

```python
    budget_series = []  # (t, F_rad, F_wind, F_SN, F_HII, F_hot) per snapshot -> budget_vs_t.csv
    # --- alpha_p: the momentum enhancement factor trinity already computes ------
    # Identity: the get_r1 root through bubble_E2P gives Pb = pdot/(4 pi R1^2) at
    # gamma = 5/3, so the force reaching the shell is F = 4 pi R2^2 Pb = pdot (R2/R1)^2.
    # Matched to Lancaster eq:alphap_shock at EQUAL BUBBLE PRESSURE (their Phot is the
    # post-shock 3 pdot/(16 pi Rf^2), so Rf = (sqrt(3)/2) R1), the 4/3 mismatch cancels
    # the 3/4 and (R2/R1)^2 IS their alpha_p to better than 1.2% for R2/R1 >~ 2.
    # See docs/dev/phii-identity/trinity_pressure_assessment.md Sec. 2.2.
    ap_max_pre = ap_handover = ap_t_hand = ap_Eb_hand = None
    ap_mom_R1_ne_R2 = 0
```

## 2. Accumulation — in the snapshot loop, immediately after `phase = d.get("current_phase")` (~line 384)

```python
            phase = d.get("current_phase")
            # --- alpha_p ------------------------------------------------------
            # Excluded inside the dt_switchon ramp window (get_bubbleParams.py:495-503),
            # where the ODE integrates a RAMPED Pb and the identity above does not hold
            # (PLAN.md Sec. 1(3) measures up to 3.31x). tSF is 0 on this grid --
            # paperII_grid_v2.param does not set it -- so the window is t <= 1e-3 Myr.
            _R1 = g(d, "R1")
            if _R1 > 0 and R2 > 0:
                if phase in _HOT_PHASES and t > 1e-3:
                    _ap = (R2 / _R1) ** 2
                    ap_max_pre = _ap if ap_max_pre is None else max(ap_max_pre, _ap)
                    if phase == "transition":
                        # overwritten each transition row -> ends up the LAST one,
                        # i.e. the value carried into the momentum phase.
                        ap_handover, ap_t_hand, ap_Eb_hand = _ap, t, g(d, "Eb")
                if phase == "momentum" and _R1 != R2:
                    ap_mom_R1_ne_R2 += 1     # must stay 0: run_momentum_phase.py:587-588
```

`_HOT_PHASES` (line 258) is already `{"energy", "implicit", "transition"}` — exactly the
pre-handover set. Reused rather than redefined.

## 3. Row assembly — after `row["t_phase_transition"] = t_phase_transition` (~line 537)

```python
    row["t_phase_transition"] = t_phase_transition
    # alpha_p = (R2/R1)^2, the force ratio. Compare against Lancaster Paper II
    # tab:cem_comp, which measures <alpha_p> = 4.57-6.82 in 3D RMHD.
    row["alpha_p_handover"] = ap_handover              # last transition row -> the gate
    row["t_alpha_p_handover"] = ap_t_hand
    row["Eb_alpha_p_handover"] = ap_Eb_hand            # should be ~ENERGY_FLOOR = 1e3
    row["alpha_p_max_pre_handover"] = ap_max_pre
    row["momentum_rows_R1_ne_R2"] = ap_mom_R1_ne_R2    # consistency check, expect 0
```

---

## The gate this discharges

Pre-registered before the patch runs on anything:

**G-A2 — is the α_p collapse physical or numerical?**
`alpha_p_handover` is the force ratio on the last transition snapshot, i.e. the value carried
into a momentum phase that applies `F = ṗ` exactly (α_p ≡ 1).

*Prediction:* `alpha_p_handover < 1.5` on essentially every run, and **its spread across the
whole v2 grid is small** — because it is set by `Eb < ENERGY_FLOOR = 1e3`
(`run_transition_phase.py:97`), not by any cloud or cluster property. `Eb_alpha_p_handover`
should cluster near 1e3, which is the check that the gate is reading the right row.

*Falsifier:* `alpha_p_handover ≥ 2` on a non-trivial fraction, **or** a spread that correlates
with a physical axis (`sfe`, `mCloud`, `nCore`, `FB_thermCoeffWind`). Either would mean the
drop to α_p = 1 is a real discontinuity in a measurable quantity rather than a smooth
handover — and with 63,360 runs the correlation is measurable rather than arguable, which is
the whole reason to do this in the survey instead of on B3M.

*Void rule:* runs that never reach `transition` return `None` and are excluded, never counted
as a confirming null.

**G-A4 — `momentum_rows_R1_ne_R2` must be 0 everywhere.** Any non-zero row falsifies the
reading that phase 2 asserts `R_f = R_w`, and with it the claim that α_p = 1 there is the
phase definition rather than an omission.

---

## Cost

Five floats per run. No new pass over `gridsweep_v2/`, no new side table, no size prompt in
`sync.sh down`. The `FORCE=1` re-reduce this needs is the same one any reducer change needs.

⚠️ Per the v2 param header and `sync.sh`'s N6 guard: reduce into `plots_v2/`, not `plots/`.
`SWEEP=$WS/paperII_grid_v2 ./sync.sh submit` already derives `DEST=plots_v2/` automatically —
do not override `DEST`.
