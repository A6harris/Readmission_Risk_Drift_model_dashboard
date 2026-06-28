"""
retrain_trigger.py — Phase 9 (stretch): close the monitoring loop.

A monitoring dashboard that only *shows* drift is half a system. This script is
the other half: it reads the drift summary, and when a monitoring window breaches
the retraining policy (the same thresholds the dashboard surfaces), it kicks off
a retrain and appends an auditable event to a log.

    python src/retrain_trigger.py                      # check all scenarios, act
    python src/retrain_trigger.py --scenario prevalence_surge
    python src/retrain_trigger.py --dry-run            # report only, never retrain

In production the retrain would run on *fresh* data; here it re-runs ``train.py``
on the same dataset to demonstrate the mechanism end to end. Every decision —
acted on or not — is logged to ``models/retrain_log.jsonl`` so there is a record
of why the model was (or wasn't) refreshed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
LOG_NAME = "retrain_log.jsonl"


def evaluate_summary(summary: dict, scenario: str | None = None) -> dict:
    """Decide whether retraining is warranted from a drift summary.

    Returns a dict with ``triggered`` (bool), the list of ``offending`` scenario
    names, and the per-scenario reasons. If ``scenario`` is given, only that
    window is considered; otherwise any scenario tripping the policy triggers.
    """
    scenarios = summary.get("scenarios", {})
    if scenario is not None:
        if scenario not in scenarios:
            raise KeyError(f"scenario '{scenario}' not in drift summary")
        names = [scenario]
    else:
        names = list(scenarios)

    offending, reasons = [], {}
    for name in names:
        sc = scenarios[name]
        if sc.get("retrain_recommended"):
            offending.append(name)
            reasons[name] = sc.get("reasons", [])
    return {"triggered": bool(offending), "offending": offending,
            "reasons": reasons, "considered": names}


def append_log(log_path: Path, event: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def run_training() -> None:
    subprocess.run([sys.executable, str(SRC / "train.py")], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--scenario", default=None,
                        help="only evaluate this monitoring window")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the decision but never retrain")
    args = parser.parse_args()

    summary_path = args.reports_dir / "drift_summary.json"
    if not summary_path.exists():
        sys.exit(f"No drift summary at {summary_path} — run `python src/drift.py` first.")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    decision = evaluate_summary(summary, args.scenario)

    now = datetime.now(timezone.utc).isoformat()
    if not decision["triggered"]:
        print(f"[retrain-trigger] no breach across {decision['considered']}; "
              f"no action.")
        action = "no_action"
    elif args.dry_run:
        print(f"[retrain-trigger] RETRAIN WARRANTED for "
              f"{decision['offending']} (dry-run, not acting).")
        for name, rs in decision["reasons"].items():
            print(f"    - {name}: {'; '.join(rs)}")
        action = "dry_run"
    else:
        print(f"[retrain-trigger] RETRAIN WARRANTED for "
              f"{decision['offending']} — retraining...")
        for name, rs in decision["reasons"].items():
            print(f"    - {name}: {'; '.join(rs)}")
        run_training()
        action = "retrained"
        print("[retrain-trigger] retrain complete; model.joblib refreshed.")

    event = {
        "timestamp": now,
        "action": action,
        "triggered": decision["triggered"],
        "offending_scenarios": decision["offending"],
        "reasons": decision["reasons"],
        "considered": decision["considered"],
    }
    log_path = args.models_dir / LOG_NAME
    append_log(log_path, event)
    print(f"[retrain-trigger] logged event ({action}) -> {log_path}")


if __name__ == "__main__":
    main()
