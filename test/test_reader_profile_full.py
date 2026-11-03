"""TrinityOutput.profile_full (docs/dev/phii-identity/weak-wind/neutral-shell/output-profiles/PLAN.md E8): the
full-resolution dump of a row, written by the writer's profile_dump_every, comes back by row index (the row's line in
dictionary.jsonl); rows without a dump give None."""
import numpy as np
import pytest

from trinity._input.dictionary import DescribedDict, DescribedItem
from trinity._output.trinity_reader import TrinityOutput


@pytest.fixture
def no_handlers(monkeypatch):
    """No atexit/signal hooks from the constructed dict (as test_dictionary_stress.py)."""
    monkeypatch.setattr(DescribedDict, "_register_crash_handlers", lambda self: None)


def test_profile_full_returns_the_dump_by_row(tmp_path, no_handlers):
    d = DescribedDict()
    d["path2output"] = DescribedItem(str(tmp_path))
    d["profile_dump_every"] = DescribedItem(2)              # rows 0 and 2 are dumped, row 1 is not
    d["t_now"] = DescribedItem(0.0)
    d["R2"] = DescribedItem(1.0)
    r = np.linspace(1.0, 0.1, 500)                           # a bubble grid, descending as the solver's
    shell_r = np.linspace(1.0, 1.2, 300)
    d["bubble_r_arr"] = DescribedItem(r)
    d["bubble_T_arr"] = DescribedItem(np.linspace(1.0e6, 1.0e4, 500))
    d["shell_r_arr"] = DescribedItem(shell_r)
    d["shell_n_arr"] = DescribedItem(np.full(300, 1.0e3))
    for i in range(3):
        d["t_now"].value = float(i)                          # the duplicate guard keys on (t_now, R2)
        d["R2"].value = 1.0 + 0.1 * i
        d.save_snapshot()
    d.flush()
    out = TrinityOutput.open(tmp_path / "dictionary.jsonl")
    assert len(out) == 3
    full = out.profile_full(0)
    assert full is not None and np.array_equal(full["shell_r_arr"], shell_r)
    assert np.array_equal(full["bubble_r_arr"], r) and float(full["t_now"]) == 0.0
    assert out.profile_full(1) is None
    full2 = out.profile_full(2)
    assert full2 is not None and float(full2["R2"]) == pytest.approx(1.2)
    assert out.profile_full(7) is None
    # the stored (simplified) radii of a dumped row all lie on its full grid
    stored = np.asarray(out[2]["shell_r_arr"], float)
    assert np.all(np.isin(stored, full2["shell_r_arr"]))


def test_profile_full_refuses_a_dump_of_another_row(tmp_path, no_handlers):
    """A dump whose t_now is not its row's gives None: files left in profiles_full/ by an earlier run into the same folder
    (a fresh run clears only dictionary.jsonl and metadata.json), and rows met through a filter()ed output, whose indices
    are no longer the files' lines. A negative index counts from the end."""
    def run(t0, dump_every):
        d = DescribedDict()
        d["path2output"] = DescribedItem(str(tmp_path))
        d["profile_dump_every"] = DescribedItem(dump_every)
        d["t_now"] = DescribedItem(t0)
        d["R2"] = DescribedItem(1.0)
        d["shell_r_arr"] = DescribedItem(np.linspace(1.0, 1.2, 300))
        d["shell_n_arr"] = DescribedItem(np.full(300, 1.0e3))
        for i in range(3):
            d["t_now"].value = t0 + i
            d["R2"].value = 1.0 + 0.1 * i
            d.save_snapshot()
        d.flush()
        return TrinityOutput.open(tmp_path / "dictionary.jsonl")

    run(0.0, 1)                                              # dumps rows 0-2
    out = run(10.0, 0)                                       # same folder, no dumps: the old files are still there
    assert (tmp_path / "profiles_full" / "row000001.npz").is_file()
    assert all(out.profile_full(i) is None for i in range(3))
    out = run(20.0, 1)
    assert float(out.profile_full(-1)["t_now"]) == 22.0
    assert out.profile_full(-4) is None and out.profile_full(3) is None
    assert out.filter(t_min=21.0).profile_full(0) is None    # filtered row 0 is the file's row 1
    assert not list((tmp_path / "profiles_full").glob("*.tmp"))


def test_a_failed_save_writes_no_dump(tmp_path, no_handlers):
    """D-OP14: the dump is written only once the row is built. A save that fails while simplifying (here a length mismatch,
    as in test_dictionary_stress's keyname test) leaves neither a row nor an orphan profiles_full/row<id>.npz."""
    d = DescribedDict()
    d["path2output"] = DescribedItem(str(tmp_path))
    d["profile_dump_every"] = DescribedItem(1)
    d["t_now"] = DescribedItem(0.0)
    d["R2"] = DescribedItem(1.0)
    d["bubble_r_arr"] = DescribedItem(np.linspace(1.0, 0.1, 20))
    d["bubble_T_arr"] = DescribedItem(np.linspace(1.0e6, 1.0e4, 19))
    with pytest.raises(ValueError, match="bubble_T_arr"):
        d.save_snapshot()
    assert not (tmp_path / "profiles_full").exists()
