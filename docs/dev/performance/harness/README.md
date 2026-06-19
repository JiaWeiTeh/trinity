# HOTPATH F1 resample — P0 capture/replay harness

> ⚠️ Point-in-time harness for the `RESAMPLE_PLAN.md` §F1 study (drop the 60k
> dense-output resample in the bubble dMdt residual). Re-verify against current
> source before trusting line refs. Results are committed as CSVs under
> `../data/`; the multi-MB state pickles are gitignored (regenerable — see below).
>
> 🟢 **(2026-06-19) F1 CLEARED.** This per-call harness is NECESSARY-NOT-SUFFICIENT — it
> measures per-call `rel_dMdt` (≤3e-6), which alone does **not** clear F1. The decider is
> **full-run equivalence** (`RESAMPLE_PLAN.md` §P5), and F1 PASSED it: `mock_hybr` (~5e-6)
> + three stiff edge cases via matched-`t` original-60k-vs-F1-coarse, worst R2/Eb ≈ 6e-6
> (`../data/f1edge_matched_comparison.csv`). **`ab_fullrun.py` is BUGGED** — both variants
> in one process → trinity global-state leakage (a false "divergence"); the correct A/B is
> **`f1_fullrun_equiv.sh`** (separate `run.py` per version), and compare at matched-`t`
> (its built-in final-state verdict false-flags when the two runs truncate at different
> `t` under the 1h cap).

## Files

| file | what it does |
|---|---|
| `residual_variants.py` | the 6 method variants. `baseline` = the unchanged production `_get_velocity_residuals` (60k dense resample); `M2000/M1000/M500/M200` = Option-(b) coarse `t_eval`; `Mnodes` = no `t_eval` (adaptive nodes only). |
| `capture_replay_bubble.py` | in-process harness: monkeypatches `get_bubbleproperties_pure`, matrix phase gate (`N_ENERGY`/`N_IMPLICIT`), runs baseline + every variant per gated bubble call, compares the 4 `BubbleProperties` outputs + times them, writes one CSV/config. Returns the baseline so the host trajectory is byte-identical. Also dumps a few view-aware state pickles to `../data/states/`. |
| `replay_from_dump.py` | offline replay: loads a state pickle (lean loader mirroring `tools/bubble_audit/audit.py:load_state`), runs all variants, prints a comparison table. |
| `aggregate_p0.py` | globs `../data/bubble_resample_*.csv` → `../data/master_p0_table.csv` + rendered markdown (worst/median `rel_*` + mean speedup per config×phase×variant). |
| `run_p0_sweep.sh` | drives `capture_replay_bubble.py` across the 6 configs at `N_ENERGY=20 N_IMPLICIT=100`, per-config wall caps, skip-if-already-≥100-implicit. |

## Reproduce

```bash
cd /home/user/trinity && pip install -e ".[dev]"   # toolchain (astropy etc.)
# single config (writes ../data/bubble_resample_mock_hybr.csv + ../data/states/*.pkl):
N_ENERGY=20 N_IMPLICIT=100 python docs/dev/performance/harness/capture_replay_bubble.py \
    docs/dev/transition/harness/mock_hybr.param
# full 6-config sweep (SLOW — degenerate configs ~45 min each):
bash docs/dev/performance/harness/run_p0_sweep.sh
# aggregate -> master table:
python docs/dev/performance/harness/aggregate_p0.py
# offline replay on a captured state:
python docs/dev/performance/harness/replay_from_dump.py docs/dev/performance/data/states/<state>.pkl
```

State pickles are **gitignored** (~3 MB each); regenerate them by running the
capture (above). The committed CSVs hold the results — read those for the
conclusions without re-running.

## P0 validation result (mock_hybr, `N_ENERGY=5 N_IMPLICIT=10`, 2026-06-18)

`data/bubble_resample_mock_hybr.csv` (90 rows = 5 energy + 10 implicit calls × 6 variants):
- **Implicit phase reached** through the `BubbleParamsView`; both phases captured.
- **Accuracy:** baseline `rel_*` = 0 (reference); worst `rel_dMdt` = **1.0e-06** (M-variants) / 2.2e-06 (Mnodes) — far under the 0.3 % G2 gate; many implicit calls bit-identical.
- **Speed (this tiny config):** baseline mean **1362 ms/call** vs variants **825–912 ms** (~1.5×). The degenerate configs (`simple_cluster`), where the 60k resample dominates more, are expected to show a larger win — that's what the full sweep measures.

Full G0 (≥4 configs at 100 implicit) is pending the sweep.
