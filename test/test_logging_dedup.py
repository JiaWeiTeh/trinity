"""DedupWarningFilter collapses identical repeated WARNING+ records, but keeps
distinct ones and anything below the threshold. Guards the large-sweep log flood
(e.g. the per-step Bonnor-Ebert 'UNSTABLE' warning)."""
import logging

from trinity._functions.logging_setup import DedupWarningFilter, setup_logging


def _rec(level, msg):
    return logging.LogRecord("t", level, __file__, 1, msg, (), None)


def test_identical_warnings_collapse_to_one():
    f = DedupWarningFilter()
    msg = "Omega=14.10 > 14.04 (critical). Sphere will be gravitationally UNSTABLE!"
    passed = [f.filter(_rec(logging.WARNING, msg)) for _ in range(1000)]
    assert passed[0] is True
    assert not any(passed[1:])            # every repeat suppressed


def test_distinct_warnings_all_pass():
    f = DedupWarningFilter()
    # varying text (a changing timestamp) must NOT collapse
    assert all(f.filter(_rec(logging.WARNING, f"unconverged at t={t}")) for t in range(50))


def test_below_min_level_never_filtered():
    f = DedupWarningFilter()
    assert all(f.filter(_rec(logging.INFO, "same info line")) for _ in range(10))


def test_per_handler_instances_are_independent():
    a, b = DedupWarningFilter(), DedupWarningFilter()
    m = "dup"
    assert a.filter(_rec(logging.WARNING, m)) is True
    assert b.filter(_rec(logging.WARNING, m)) is True   # separate state -> each shows once
    assert a.filter(_rec(logging.WARNING, m)) is False


def test_end_to_end_file_log_has_each_warning_once(tmp_path):
    setup_logging(log_level="INFO", console_output=False, file_output=True,
                  log_file_path=tmp_path, log_file_name="t.log", use_colors=False)
    log = logging.getLogger("trinity.cloud_properties.bonnorEbertSphere")
    for _ in range(500):
        log.warning("Omega=14.10 > 14.04 (critical). Sphere will be gravitationally UNSTABLE!")
    log.warning("a different warning")
    for h in logging.getLogger().handlers:
        h.flush()
    text = (tmp_path / "t.log").read_text()
    assert text.count("gravitationally UNSTABLE") == 1
    assert text.count("a different warning") == 1
