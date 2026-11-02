from __future__ import annotations

import json
from pathlib import Path

from evaluation.reporting import generate_experiment_report
from utils.logging import JsonlLogger


def test_reporting_writes_real_charts(tmp_path: Path) -> None:
    config = {
        "paths": {
            "logs": str(tmp_path / "logs"),
            "results": str(tmp_path / "results"),
            "checkpoints": str(tmp_path / "checkpoints"),
        }
    }
    logs = tmp_path / "logs" / "demo"
    results = tmp_path / "results" / "demo"
    checkpoints = tmp_path / "checkpoints" / "demo"
    logs.mkdir(parents=True)
    results.mkdir(parents=True)
    checkpoints.mkdir(parents=True)
    logger = JsonlLogger(logs / "metrics.jsonl")
    for iteration in range(1, 4):
        logger.write(
            {
                "iteration": iteration,
                "policy_loss": 2.0 / iteration,
                "value_loss": 0.8 / iteration,
                "total_loss": 2.8 / iteration,
                "win_rate_vs_random": 0.4 + iteration * 0.1,
                "candidate_score_vs_best": 0.48 + iteration * 0.03,
            }
        )
    history = [
        {
            "iteration": iteration,
            "type": "candidate_vs_best",
            "wins": 10 + iteration,
            "losses": 8,
            "draws": 2,
            "games": 20,
            "win_rate": (10 + iteration) / 20,
            "score": (11 + iteration) / 20,
        }
        for iteration in range(1, 4)
    ]
    (results / "evaluations.json").write_text(json.dumps(history), encoding="utf-8")
    (checkpoints / "promotions.json").write_text(
        json.dumps([{"iteration": 2, "score": 0.6}]), encoding="utf-8"
    )
    paths = generate_experiment_report(config, "demo")
    names = {path.name for path in paths}
    assert "loss_curves.png" in names
    assert "win_rates.png" in names
    assert "latest_evaluation.png" in names
    assert "promotions.png" in names
    assert "report_summary.json" in names
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
