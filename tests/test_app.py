"""Smoke test for the Streamlit dashboard.

Runs the app headlessly via Streamlit's AppTest and asserts it executes without
raising. This passes even on a fresh checkout where generated figures/models are
absent, because the app degrades gracefully (it shows "run this step" messages
rather than crashing) — which is exactly the behavior we want to guard.
"""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = "src/monitor_app.py"


def test_app_runs_without_exception():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert at.exception is None or len(at.exception) == 0
    assert len(at.error) == 0
    # The three top-level tabs should always render.
    assert len(at.tabs) == 3


def test_monitoring_scenarios_switch_cleanly():
    at = AppTest.from_file(APP, default_timeout=120).run()
    if not at.selectbox:
        pytest.skip("drift_summary.json absent; monitoring selectbox not rendered")
    for scenario in ["baseline", "pipeline_break", "prevalence_surge"]:
        at.selectbox[0].select(scenario).run()
        assert at.exception is None or len(at.exception) == 0
