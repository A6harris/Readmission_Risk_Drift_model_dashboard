"""Tests for the retrain decision + logging in src/retrain_trigger.py."""

import json

import pytest

from src.retrain_trigger import append_log, evaluate_summary


def _summary():
    return {
        "scenarios": {
            "baseline": {"retrain_recommended": False, "reasons": []},
            "age_shift": {"retrain_recommended": False, "reasons": []},
            "pipeline_break": {"retrain_recommended": True,
                               "reasons": ["AUROC fell 0.043 vs reference"]},
            "prevalence_surge": {"retrain_recommended": True,
                                 "reasons": ["Brier rose 0.108 vs reference"]},
        }
    }


def test_triggers_when_any_scenario_breaches():
    d = evaluate_summary(_summary())
    assert d["triggered"] is True
    assert set(d["offending"]) == {"pipeline_break", "prevalence_surge"}
    assert "AUROC" in d["reasons"]["pipeline_break"][0]


def test_no_trigger_for_healthy_scenario():
    d = evaluate_summary(_summary(), scenario="baseline")
    assert d["triggered"] is False
    assert d["offending"] == []
    assert d["considered"] == ["baseline"]


def test_single_scenario_selection():
    d = evaluate_summary(_summary(), scenario="prevalence_surge")
    assert d["triggered"] is True
    assert d["offending"] == ["prevalence_surge"]


def test_unknown_scenario_raises():
    with pytest.raises(KeyError):
        evaluate_summary(_summary(), scenario="does_not_exist")


def test_append_log_writes_jsonl(tmp_path):
    log = tmp_path / "retrain_log.jsonl"
    append_log(log, {"action": "no_action", "triggered": False})
    append_log(log, {"action": "retrained", "triggered": True})
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["action"] == "retrained"
