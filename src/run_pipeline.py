"""
run_pipeline.py — run the whole project end to end, in order.

    python src/run_pipeline.py            # full pipeline
    python src/run_pipeline.py --from train   # skip data_prep (reuse processed)
    python src/run_pipeline.py --only drift   # just one step

Each phase is a standalone script; this simply runs them in dependency order
with the current Python interpreter so it works the same on Windows, macOS,
Linux, and inside Docker. Launch the dashboard separately with:

    streamlit run src/monitor_app.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent

# Ordered phases. monitor_app is intentionally excluded — it's a server, not a
# batch step.
STEPS = [
    ("data_prep", "Phase 1 — download + clean + split"),
    ("train", "Phase 2 — train LR + XGBoost, serialize best"),
    ("evaluate", "Phase 3 — calibration + net benefit"),
    ("fairness", "Phase 4 — subgroup fairness audit"),
    ("explain", "Phase 5 — SHAP explainability"),
    ("drift", "Phase 6 — drift simulation + Evidently"),
]


def run_step(name: str) -> None:
    script = SRC / f"{name}.py"
    print(f"\n{'=' * 70}\n>> {name}\n{'=' * 70}", flush=True)
    t0 = time.time()
    subprocess.run([sys.executable, str(script)], check=True)
    print(f"[ok] {name} done in {time.time() - t0:.1f}s", flush=True)


def main() -> None:
    names = [s[0] for s in STEPS]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", choices=names,
                        help="start from this phase (skip earlier ones)")
    parser.add_argument("--only", choices=names,
                        help="run only this phase")
    args = parser.parse_args()

    if args.only:
        selected = [args.only]
    elif args.start:
        selected = names[names.index(args.start):]
    else:
        selected = names

    print("Pipeline plan: " + " -> ".join(selected))
    t0 = time.time()
    for name in selected:
        run_step(name)
    print(f"\nAll done in {time.time() - t0:.1f}s. "
          f"Launch the dashboard: streamlit run src/monitor_app.py")


if __name__ == "__main__":
    main()
