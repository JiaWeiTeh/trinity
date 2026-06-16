# T_init sensitivity sweep

A robust, deterministic test of whether the bubble-structure boundary
temperature `T_init` (the `_T_INIT_BOUNDARY = 3e4` K constant in
`bubble_structure/bubble_luminosity.py`) is a *relabel-only numerical knob*,
i.e. whether `L_total` and its components are invariant to the choice of anchor.

See `docs/dev/misc/tinit-sensitivity.md` for the findings.

## Design (why it cannot pass by coincidence)

The harness runs the **full** production pipeline (`get_bubbleproperties_pure`:
R1/Pb → fsolve `dMdt` → initial conditions → ODE → luminosity) on a captured
bubble state, varying only `T_init` via the single coupled knob
`bl._T_INIT_BOUNDARY` (which feeds the integration anchor *and* the fsolve
rejection floor *and* its penalty — all three move together). Five gates:

1. **Fidelity** — at the baseline `T_init=3e4`, the re-solved `dMdt` reproduces
   the dumped converged value (`rel < 1e-3`). States are dumped by *current*
   code, so this is a true ground truth; `tools/bubble_audit/audit.py`
   separately confirms bit-exact `T_array` reproduction.
2. **Determinism** — across `K` **separate, thread-pinned** processes, every
   `(state, T_init)` `L_total` is **bit-identical** (`float.hex`). A result that
   is not bit-reproducible is treated as invalid, not a pass. Separate processes
   (not an in-process loop) are required to catch cross-process BLAS-threading
   FP nondeterminism.
3. **Contrast** — across `K` **unpinned** processes, report whether `L_total`
   varies. Demonstrates the determinism in (2) is earned by pinning, not luck.
   Observational (a single-core box may not vary).
4. **Sensitivity** — per state, `|L(T_init) − L(3e4)| / L(3e4) ≤ 1%` across the
   grid. The actual physics question.
5. **Robustness** — every crash / `MonotonicError` is counted per cell and
   printed; a crash is never silently a pass.

## Regenerating the captured states

States are not committed (≈5 MB each). Regenerate with the smoke param
(~2.5 min, current code), pinned for reproducibility:

```bash
mkdir -p /tmp/tinit && cd /tmp/tinit
printf 'mCloud 1e5\nsfe 0.3\nstop_t 1e-4\nmodel_name tinit_dump\n' > smoke.param
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  TRINITY_BUBBLE_STATE_DUMP=12 python /path/to/run.py smoke.param
# states land in outputs/tinit_dump/bubble_state/*.pkl
```

These states are early energy-phase, all at `(β+δ)=0.629` (the dip-prone high
value). For `(β+δ)` breadth you need later-phase states (a longer run).

## Running

```bash
# all gates, K=5 repeats, over a directory of states:
python tools/tinit_sweep/run_sweep.py <states_dir> --k 5
# per-T_init detail table (where the sensitivity lives):
OMP_NUM_THREADS=1 ... python tools/tinit_sweep/profile_tinit.py <states_dir>
```
