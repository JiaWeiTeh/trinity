"""The committed bench7 param set stays exactly what its builder emits, and stays single-knob.

Covers docs/dev/transition/pdv-trigger/runs/make_kappa_reopen_params.py — the f_kappa re-open
campaign (KAPPA_REOPEN_PLAN.md K1-K4). Two reasons this is worth a test rather than a re-read:

1. The campaign's reduce is ONE-SHOT (gpfs is cleaned; the raw arms do not come back), so a param
   that drifts from its builder is not recoverable after submission.
2. Every prediction in the plan assumes SINGLE-KNOB arms — the f_kappa arms at cooling_boost_mode
   none / f_A 1, the f_mix arms at f_kappa 1 — and P4 assumes the K3 pairs differ in nothing but
   their names, since it is decided by a hash of the reduced trajectories.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

RUNS = Path(__file__).resolve().parent.parent / "docs/dev/transition/pdv-trigger/runs"
PARAMS = RUNS / "params" / "bench7"

# docs/dev is untracked (local-only, see .gitignore) as of `a32b098`: absent in a
# fresh clone and in CI. Every test here reads PARAMS/RUNS -- and without them the
# parametrised arm test silently collects zero cases, which is worse than a skip.
if not PARAMS.is_dir():
    pytest.skip(
        "docs/dev is untracked (local-only); bench7 param set unavailable",
        allow_module_level=True,
    )

PHASE_COUNTS = {"k1_": 54, "k1b_": 20, "k2_": 66, "k3_": 10, "k4_": 24}
KNOBS = ("cooling_boost_kappa", "cooling_boost_fA", "cooling_boost_mode")


def _kv(path):
    out = {}
    for line in path.read_text().splitlines():
        key, _, val = line.partition(" ")
        out[key] = val.strip()
    return out


@pytest.fixture(scope="module")
def committed():
    files = sorted(PARAMS.glob("*.param"))
    assert files, f"no committed params in {PARAMS} — run make_kappa_reopen_params.py"
    return {p.stem: p.read_text() for p in files}


def test_params_regenerate_byte_identically(tmp_path, committed):
    """The builder is the source of truth: re-emitting must reproduce every committed byte."""
    sys.path.insert(0, str(RUNS))
    try:
        spec = importlib.util.spec_from_file_location(
            "make_kappa_reopen_params", RUNS / "make_kappa_reopen_params.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.OUT = tmp_path
        mod.main()
    finally:
        sys.path.remove(str(RUNS))

    fresh = {p.stem: p.read_text() for p in tmp_path.glob("*.param")}
    assert fresh == committed


def test_phase_prefixes_and_counts(committed):
    """K1/K1b/K2/K3/K4 all share one array, so the phase split is only the filename prefix."""
    for prefix, n in PHASE_COUNTS.items():
        # k1_ must not swallow k1b_ — longest matching prefix wins.
        got = [
            s
            for s in committed
            if max((p for p in PHASE_COUNTS if s.startswith(p)), key=len) == prefix
        ]
        assert len(got) == n, f"{prefix}: {len(got)} arms, expected {n}"
    assert len(committed) == sum(PHASE_COUNTS.values())


@pytest.mark.parametrize("stem", sorted(p.stem for p in PARAMS.glob("*.param")))
def test_arm_is_single_knob_and_protocol_compliant(stem):
    kv = _kv(PARAMS / f"{stem}.param")
    assert kv["stop_t"] == "5", "standing rule 1: every arm runs to 5 Myr"
    assert kv["model_name"] == stem
    assert (
        kv["path2output"] == f"outputs/bench7/{stem}"
    ), "run_bench7.sbatch keys .exit_code off this"

    active = [k for k in KNOBS if k in kv]
    assert len(active) == 1, f"{stem} sets {active} — arms are single-knob by construction"
    if "cooling_boost_mode" in kv:
        assert kv["cooling_boost_mode"] == "multiplier" and "cooling_boost_fmix" in kv
    else:
        assert "cooling_boost_fmix" not in kv

    # diag arms are the uncensored-theta(t) half of the two-arm protocol; prod arms run the live
    # cooling_balance trigger, which is what produces the fire map.
    assert ("transition_trigger" in kv) == stem.endswith("_diag")


def test_k3_pairs_differ_only_in_their_names():
    """P4 is decided by hashing the reduced trajectories, so the pair must be identical physics."""
    pairs = {p.stem[:-2] for p in PARAMS.glob("k3_*_a.param")}
    assert len(pairs) == PHASE_COUNTS["k3_"] // 2
    for base in pairs:
        a, b = (_kv(PARAMS / f"{base}_{r}.param") for r in "ab")
        assert a.keys() == b.keys()
        assert {k: v for k, v in a.items() if k not in ("model_name", "path2output")} == {
            k: v for k, v in b.items() if k not in ("model_name", "path2output")
        }


@pytest.mark.parametrize(
    "bench",
    [
        "bench1_m5e4_r20",
        "bench2_m1e5_r10",
        "bench3_m1e5_r5",
        "bench4_m1e5_r2p5",
        "bench5_m5e5_r2p5",
    ],
)
def test_bench_arms_share_the_bench5_baseline(bench):
    """G2 compares each new arm against its bench5 ``__none`` sibling — same cloud, or no baseline."""
    ignore = ("model_name", "path2output", "transition_trigger")
    base = {
        k: v
        for k, v in _kv(RUNS / "params" / "bench5" / f"{bench}__none.param").items()
        if k not in ignore
    }
    arms = [p for p in PARAMS.glob(f"*_{bench}__*.param")]
    assert arms, f"no bench7 arm for {bench}"
    for arm in arms:
        got = {
            k: v
            for k, v in _kv(arm).items()
            if k not in ignore and not k.startswith("cooling_boost_")
        }
        assert got == base, f"{arm.stem} does not sit on the bench5 {bench} cloud"
