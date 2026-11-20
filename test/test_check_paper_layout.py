"""tools/check_paper_layout.py on a fake paper/ folder (the real ones are local-only)."""

from tools.check_paper_layout import check, main


def _project(root, name, files=(), dirs=()):
    proj = root / name
    proj.mkdir(parents=True)
    for d in dirs:
        (proj / d).mkdir(parents=True, exist_ok=True)
    for f, text in dict(files).items():
        (proj / f).parent.mkdir(parents=True, exist_ok=True)
        (proj / f).write_text(text)
    return proj


def test_a_standard_project_is_clean(tmp_path):
    proj = _project(
        tmp_path / "paper",
        "good",
        files={
            "PLAN.md": "see paper/good/plots/a.pdf\n",
            "plots/a.pdf": "%PDF",
            "prompts/PROMPT_continue.md": "read paper/good/PLAN.md\n",
        },
        dirs=["data/v1/big_runs", "harness", ".claude"],
    )
    assert check(proj) == []
    assert main(["--root", str(tmp_path / "paper"), "good"]) == 0


def test_clutter_synonyms_stale_paths_and_the_marker_are_reported(tmp_path):
    proj = _project(
        tmp_path / "paper",
        "bad",
        files={
            "PLAN.md": "run paper/bad/figures/old.py\n",
            "make_fig.py": "",
            "x.param": "",
            "REORG_IN_PROGRESS.md": "",
        },
        dirs=["figures", "plots_v6", "to-be-removed", "misc"],
    )
    issues = "\n".join(check(proj))
    for expected in (
        "REORG_IN_PROGRESS.md",
        "make_fig.py: a loose file",
        "x.param: a loose file",
        "figures/: a synonym",
        "plots_v6/",
        "to-be-removed/: a synonym",
        "misc/: not a standard folder",
        "PLAN.md names paper/bad/figures/old.py",
    ):
        assert expected in issues
    assert main(["--root", str(tmp_path / "paper"), "bad"]) == 1


def test_light_folders_and_recorded_exceptions(tmp_path):
    root = tmp_path / "paper"
    light = _project(
        root, "janice", files={"README.md": "", "src/a.py": ""}, dirs=["out", "correspondence"]
    )
    phangs = _project(root, "trinity-phangs-I", files={"PLAN.md": "", "digest_catalogue.py": ""})
    closed = _project(root, "shellSSC6", files={"README.md": "", "sync.sh": ""})
    assert check(light) == [] and check(phangs) == [] and check(closed) == []
    assert (
        check(_project(root, "pedrini", files={"script.py": ""})) == []
    )  # light: loose files are fine
    assert any("to-be-removed" in i for i in check(_project(root, "hamed", dirs=["to-be-removed"])))
    made = _project(root, "made", files={"PLAN.md": "sed x > paper/made/param/new.param\n"})
    assert check(made) == []  # a redirect target is created by the command, not stale
