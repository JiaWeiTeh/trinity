"""Site-profile loading (trinity/_input/cluster_profile.py) — workstream C.

The profile holds the per-user HPC scheduler bits (partition/time/mem/account/
export/throttle/chunk + an env prologue) so they are set ONCE instead of
hand-edited into every emitted sbatch. A missing file must be a no-op.
"""
from __future__ import annotations

import textwrap

from trinity._input.cluster_profile import ClusterProfile, load_profile


def _write(tmp_path, text, name="cluster.ini"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(text), encoding="utf-8")
    return p


def test_absent_profile_is_empty(tmp_path):
    prof = load_profile(tmp_path / "does_not_exist.ini")
    assert prof == ClusterProfile()
    assert prof.prologue == "" and prof.partition is None


def test_loads_sbatch_and_submit_fields(tmp_path):
    p = _write(tmp_path, """
        [sbatch]
        partition = cpu-single
        time = 02:00:00
        mem = 2G
        export = NONE
        [submit]
        throttle = 150
        chunk = 880
    """)
    prof = load_profile(p)
    assert prof.partition == "cpu-single"
    assert prof.time == "02:00:00"
    assert prof.mem == "2G"
    assert prof.export == "NONE"
    assert prof.throttle == 150
    assert prof.chunk == 880
    assert prof.account is None
    assert prof.source == str(p)


def test_chunk_auto_and_bad_int_become_none(tmp_path):
    p = _write(tmp_path, """
        [submit]
        throttle = auto
        chunk = notanumber
    """)
    prof = load_profile(p)
    assert prof.throttle is None
    assert prof.chunk is None


def test_percent_in_value_does_not_break_parsing(tmp_path):
    """Interpolation is disabled so a literal '%' is fine."""
    p = _write(tmp_path, """
        [env]
        prologue = echo 100%% done
    """)
    # Either raw or unescaped is acceptable; the point is it must not raise.
    assert "done" in load_profile(p).prologue


def test_prologue_file_read_relative_to_profile_dir(tmp_path):
    (tmp_path / "prologue.sh").write_text(
        "module load devel/miniforge\nconda activate trinity\n"
    )
    p = _write(tmp_path, """
        [env]
        prologue_file = prologue.sh
    """)
    prof = load_profile(p)
    assert "module load devel/miniforge" in prof.prologue
    assert "conda activate trinity" in prof.prologue


def test_inline_multiline_prologue(tmp_path):
    p = _write(tmp_path, """
        [env]
        prologue = module load devel/miniforge
            conda activate trinity
    """)
    prof = load_profile(p)
    assert "module load devel/miniforge" in prof.prologue
    assert "conda activate trinity" in prof.prologue


def test_prologue_file_wins_over_inline(tmp_path):
    (tmp_path / "pro.sh").write_text("FROM_FILE=1\n")
    _write(tmp_path, """
        [env]
        prologue_file = pro.sh
        prologue = FROM_INLINE=1
    """)
    assert load_profile(tmp_path / "cluster.ini").prologue == "FROM_FILE=1"


def test_env_var_discovery(tmp_path, monkeypatch):
    p = _write(tmp_path, "[sbatch]\npartition = gpu\n")
    monkeypatch.setenv("TRINITY_CLUSTER_PROFILE", str(p))
    assert load_profile().partition == "gpu"
