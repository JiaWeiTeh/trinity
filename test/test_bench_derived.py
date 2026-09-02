"""The on-cluster distilled scalars equal the canonical laptop-side metrics.

``harvest_bench5.py --derived`` computes Θ_cum and the solved/stale split ON the cluster during
`reduce`, so a 500+-arm campaign (bench8/f_area, F_AREA_PLAN.md §9a) is answerable from one summary
CSV instead of from N trajectory files. It cannot import the canonical implementations — both live
in ``data/`` modules that import matplotlib at module level, and the reduce step must stay
dependency-light on the cluster — so the arithmetic is duplicated. This pins the duplication:

  * ``derived`` vs ``make_bench5_analysis.theta_cum_prefire``   (Θ_cum, effective-loss numerator)
  * ``derived`` vs ``make_bench_stale_segments.decompose``      (stale/solved split, θ_max_solved)

A drift here would silently republish the FINDINGS §12 conclusion (f_A is the best single knob on
both axes) off numbers that no longer match the metric it was decided on. ``--traj-bundle`` is
covered too: the bundle must carry byte-identical values to the per-arm files it replaces, since
after bench8 the per-arm files never come down at all.
"""

import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PDV = ROOT / "docs/dev/transition/pdv-trigger"

# docs/dev is untracked (local-only, see .gitignore) as of `a32b098`: absent in a
# fresh clone and in CI. Every test here reads that tree, so skip the module.
if not all((PDV / rel).is_file() for rel in (
        "runs/harvest_bench5.py", "data/make_bench5_analysis.py",
        "data/make_bench_stale_segments.py", "data/read_bundle.py")):
    pytest.skip(
        "docs/dev is untracked (local-only); pdv-trigger harness/data unavailable",
        allow_module_level=True,
    )


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def hb():
    return _load(PDV / "runs/harvest_bench5.py", "harvest_bench5")


@pytest.fixture(scope="module")
def canon():
    return (_load(PDV / "data/make_bench5_analysis.py", "make_bench5_analysis"),
            _load(PDV / "data/make_bench_stale_segments.py", "make_bench_stale_segments"))


@pytest.fixture(scope="module")
def rb():
    return _load(PDV / "data/read_bundle.py", "read_bundle")


# A trajectory with the shape the metric actually has to survive: a rising θ, a FROZEN-L_cool
# stretch (rows 3-4 repeat L_cool — the β–δ no-root staleness of FINDINGS §12) whose θ keeps
# climbing because L_mech still evolves, and a null R2. Values are the run's own units.
TRAJ = [
    #  t_now, theta,  Lcool,   Lleak,  Lmech,   R2
    [0.10, 0.20, 2.0e37, 1.0e36, 1.05e38, 3.0],
    [0.22, 0.41, 4.3e37, 1.2e36, 1.08e38, 5.2],
    [0.55, 0.63, 6.9e37, 1.5e36, 1.11e38, 8.1],
    [0.91, 0.88, 9.9e37, 1.5e36, 1.13e38, 11.4],   # last SOLVED row (its L_cool did move)
    [1.40, 0.96, 9.9e37, 1.5e36, 1.16e38, 14.9],   # STALE (L_cool frozen), and it SETS theta_max
    [2.10, 0.71, 8.2e37, 1.7e36, 1.19e38, None],
]


def _roundtrip(hb, tmp_path, rows, extra=()):
    """Write rows through write_traj and read them back as the analysis scripts do."""
    run = tmp_path / "arm_x"
    run.mkdir(exist_ok=True)
    (run / "dictionary.jsonl").write_text("")
    hb.write_traj(run, tmp_path / "traj", extra, rows=rows)
    path = tmp_path / "traj" / "arm_x.csv"
    with open(path) as fh:
        return list(csv.DictReader(x for x in fh if not x.lstrip().startswith("#")))


def test_theta_cum_matches_the_canonical_metric(hb, canon, tmp_path):
    """Θ_cum, its raw twin, the window end and the leak fraction all agree to ~ULP."""
    b5, _ = canon
    got = hb.derived(TRAJ)
    want = b5.theta_cum_prefire(_roundtrip(hb, tmp_path, TRAJ))
    assert got["theta_cum"] == pytest.approx(want[0], rel=1e-12)
    assert got["theta_cum_raw"] == pytest.approx(want[1], rel=1e-12)
    assert got["t_window_end"] == pytest.approx(want[2], rel=1e-12)
    assert got["leak_frac"] == pytest.approx(want[3], rel=1e-12)


def test_stale_split_matches_the_canonical_decompose(hb, canon, tmp_path):
    """The §12 convention: stale rows are excluded, and θ_max_solved is the trigger metric."""
    _, seg = canon
    got = hb.derived(TRAJ)
    n, n_stale, tfrac, th_stale, th_solved, max_is_stale, max_solved = seg.decompose(
        _roundtrip(hb, tmp_path, TRAJ))
    assert (got["n_rows"], got["n_stale"]) == (n, n_stale)
    assert got["stale_time_frac"] == pytest.approx(tfrac, rel=1e-12)
    assert got["theta_cum_stale"] == pytest.approx(th_stale, rel=1e-12)
    assert got["theta_cum_solved"] == pytest.approx(th_solved, rel=1e-12)
    assert got["theta_max_is_stale"] is max_is_stale is True      # the fixture is built to trip it
    assert got["theta_max_solved"] == pytest.approx(max_solved, rel=1e-12)


def test_theta_max_solved_excludes_the_stale_peak(hb):
    """The whole point of §12: the frozen-row peak (0.96) must not be what the trigger metric sees.

    Only the SECOND row of a repeated-L_cool pair is stale — the segment across which the solver
    found no root — so the solved max is the 0.88 row, not the 0.71 tail.
    """
    got = hb.derived(TRAJ)
    assert max(r[1] for r in TRAJ) == 0.96          # the raw peak sits on a stale row
    assert got["theta_max_solved"] == pytest.approx(0.88)


def test_degenerate_trajectories_do_not_raise(hb):
    """A wall-killed arm can reduce to 0 or 1 rows — every scalar goes null, nothing explodes."""
    for rows in ([], TRAJ[:1]):
        assert hb.derived(rows) == dict.fromkeys(hb.DERIVED_COLS)


def _fake_run(d, name, rows):
    """A run dir whose dictionary.jsonl replays ``rows`` as accepted implicit snapshots."""
    run = d / name
    run.mkdir(parents=True)
    with (run / "dictionary.jsonl").open("w") as fh:
        for t, th, lc, lk, lm, r2 in rows:
            fh.write(json.dumps({"current_phase": "implicit", "t_now": t, "bubble_Lloss": th * lm,
                                 "bubble_LTotal": lc, "bubble_Leak": lk, "Lmech_total": lm,
                                 "R2": r2, "Pb": 1.5e-9}) + "\n")
    return run


def test_bundle_carries_the_same_values_as_the_per_arm_files(hb, rb, tmp_path):
    """After bench8 the per-arm CSVs never come down, so the bundle must lose nothing."""
    for nm, rows in (("arm_a", TRAJ), ("arm_b", TRAJ[:4])):
        _fake_run(tmp_path / "runs", nm, rows)
    summary, bundle, traj_dir = tmp_path / "s.csv", tmp_path / "b.csv", tmp_path / "t"
    hb.main([str(tmp_path / "runs" / "arm_a"), str(tmp_path / "runs" / "arm_b"),
             "--csv", str(summary), "--derived",
             "--traj-bundle", str(bundle), "--traj-dir", str(traj_dir),
             "--extra-cols", "Pb"])

    loaded = rb.load(bundle)
    assert set(loaded) == {"arm_a", "arm_b"}
    for name, got in loaded.items():
        with open(traj_dir / f"{name}.csv") as fh:
            per_arm = list(csv.DictReader(x for x in fh if not x.lstrip().startswith("#")))
        assert got == per_arm                        # identical strings, not merely close floats
        assert "Pb" in got[0]                        # --extra-cols survives into the bundle


def test_summary_carries_the_derived_columns(hb, tmp_path):
    """The headline read is the summary alone — the distilled scalars must be IN it."""
    _fake_run(tmp_path / "runs", "arm_a", TRAJ)
    summary = tmp_path / "s.csv"
    hb.main([str(tmp_path / "runs" / "arm_a"), "--csv", str(summary), "--derived"])
    with open(summary) as fh:
        rows = list(csv.DictReader(x for x in fh if not x.lstrip().startswith("#")))
    assert len(rows) == 1
    assert set(hb.DERIVED_COLS) <= set(rows[0])
    assert float(rows[0]["theta_max_solved"]) == pytest.approx(0.88)
    assert float(rows[0]["theta_max"]) == pytest.approx(0.96)    # unfiltered stays, for the audit


def test_derived_reproduces_the_published_bench7_record(hb):
    """Replay all 173 committed bench7 trajectories — the on-cluster scalars must match the record.

    The synthetic fixtures above pin the arithmetic; this pins it against the DATA the §12
    conclusion was actually drawn from (data/bench_stale_segments.csv, written with %.4f — so the
    bar is that half-ulp, 5e-5, not machine epsilon). Runs offline from committed CSVs: no sim.
    """
    traj_dir, record = PDV / "runs/data/bench7_traj", PDV / "data/bench_stale_segments.csv"
    if not traj_dir.is_dir() or not record.exists():
        pytest.skip("bench7 harvest not present")

    def read(p):
        with open(p) as fh:
            return list(csv.DictReader(x for x in fh if not x.lstrip().startswith("#")))

    rec = {r["run_name"]: r for r in read(record)}
    pairs = [("theta_max_solved", "theta_max_solved"), ("theta_cum_solved", "theta_cum_from_solved"),
             ("theta_cum_stale", "theta_cum_from_stale"), ("stale_time_frac", "stale_time_frac")]
    compared = 0
    for p in sorted(traj_dir.glob("*.csv")):
        rows, r = read(p), rec.get(p.stem)
        if len(rows) < 2 or r is None:
            continue
        got = hb.derived([[float(x["t_now"]), float(x["theta"]), float(x["Lcool"]),
                           float(x["Lleak"]), float(x["Lmech"]), None] for x in rows])
        assert (got["n_rows"], got["n_stale"]) == (int(r["n_rows"]), int(r["n_stale"])), p.stem
        for got_key, rec_key in pairs:
            if r.get(rec_key):
                assert got[got_key] == pytest.approx(float(r[rec_key]), abs=5e-5), f"{p.stem}/{got_key}"
        compared += 1
    assert compared > 150, f"expected the full bench7 harvest, compared only {compared}"


def test_derived_is_opt_in(hb, tmp_path):
    """Without --derived the summary keeps exactly its published columns (frozen campaigns)."""
    _fake_run(tmp_path / "runs", "arm_a", TRAJ)
    summary = tmp_path / "s.csv"
    hb.main([str(tmp_path / "runs" / "arm_a"), "--csv", str(summary)])
    with open(summary) as fh:
        header = list(csv.DictReader(x for x in fh if not x.lstrip().startswith("#")))[0]
    assert set(header) & set(hb.DERIVED_COLS) == set()
