"""Generate Matplotlib charts from a completed experiment's real artifacts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.reporting import generate_before_after_chart, generate_experiment_report
from utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fast_demo.yaml")
    parser.add_argument("--experiment", default="demo")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = generate_experiment_report(config, args.experiment)
    if not paths:
        print("No metrics found; run training first.")
        return 1
    before_after = generate_before_after_chart(config)
    if before_after:
        paths.append(before_after)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
