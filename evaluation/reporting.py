"""Matplotlib report generation from actual training/evaluation artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.config import log_dir, resolve_path, results_dir, training_dir
from utils.logging import read_json, read_jsonl


BG = "#111927"
PANEL = "#172233"
TEXT = "#e9f0f8"
DIM = "#90a1b8"
ACCENT = "#3acae8"
GOLD = "#f6be5c"
GREEN = "#4bcd8b"
RED = "#f05658"
PURPLE = "#977ef0"


def _style_axis(axis) -> None:
    axis.set_facecolor(PANEL)
    axis.tick_params(colors=DIM, labelsize=8)
    for spine in axis.spines.values():
        spine.set_color("#2f3f57")
    axis.grid(color="#2a394f", alpha=0.55, linewidth=0.7)
    axis.set_axisbelow(True)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(BG)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=BG)
    plt.close(fig)
    return path


def generate_experiment_report(
    config: dict,
    experiment: str,
    output_dir: Optional[str | Path] = None,
) -> List[Path]:
    metrics_path = log_dir(config, experiment) / "metrics.jsonl"
    history_path = results_dir(config, experiment) / "evaluations.json"
    promotions_path = training_dir(config, experiment) / "promotions.json"
    metrics = read_jsonl(metrics_path)
    history = read_json(history_path, []) or []
    promotions = read_json(promotions_path, []) or []
    output = Path(output_dir) if output_dir else results_dir(config, experiment) / "plots"
    output.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    if not metrics:
        return outputs

    iterations = [int(m["iteration"]) for m in metrics]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _style_axis(ax)
    for key, color, label in (
        ("policy_loss", ACCENT, "policy loss"),
        ("value_loss", PURPLE, "value loss"),
        ("total_loss", GOLD, "total loss"),
    ):
        values = [m.get(key) for m in metrics]
        if any(v is not None and np.isfinite(v) for v in values):
            ax.plot(iterations, values, color=color, linewidth=2, marker="o", markersize=3, label=label)
    ax.set_title("Training losses", color=TEXT, fontsize=13, loc="left")
    ax.set_xlabel("iteration", color=DIM)
    ax.set_ylabel("loss", color=DIM)
    ax.legend(facecolor=PANEL, edgecolor="#2f3f57", labelcolor=TEXT, fontsize=8)
    outputs.append(_save(fig, output / "loss_curves.png"))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _style_axis(ax)
    candidate = [(h.get("iteration"), h.get("win_rate")) for h in history if h.get("type") == "candidate_vs_best"]
    random_results = [(h.get("iteration"), h.get("win_rate")) for h in history if h.get("type") == "best_vs_random"]
    tactical = [(h.get("iteration"), h.get("win_rate")) for h in history if h.get("type") == "best_vs_tactical"]
    if candidate:
        xs, ys = zip(*candidate)
        ax.plot(xs, ys, color=ACCENT, marker="o", linewidth=2, label="candidate win rate vs best")
    if random_results:
        xs, ys = zip(*random_results)
        ax.plot(xs, ys, color=GREEN, marker="s", linewidth=2, label="best win rate vs random")
    if tactical:
        xs, ys = zip(*tactical)
        ax.plot(xs, ys, color=PURPLE, marker="^", linewidth=2, label="best win rate vs tactical")
    ax.set_ylim(0, 1)
    ax.set_title("Arena win rates", color=TEXT, fontsize=13, loc="left")
    ax.set_xlabel("iteration", color=DIM)
    ax.set_ylabel("win rate", color=DIM)
    ax.legend(facecolor=PANEL, edgecolor="#2f3f57", labelcolor=TEXT, fontsize=8)
    outputs.append(_save(fig, output / "win_rates.png"))

    candidate_records = [h for h in history if h.get("type") == "candidate_vs_best"]
    if candidate_records:
        latest = candidate_records[-1]
        fig, ax = plt.subplots(figsize=(6, 4))
        _style_axis(ax)
        labels = ["wins", "losses", "draws"]
        values = [latest.get("wins", 0), latest.get("losses", 0), latest.get("draws", 0)]
        bars = ax.bar(labels, values, color=[GREEN, RED, GOLD])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, str(value), ha="center", color=TEXT)
        ax.set_title(f"Candidate vs best - iteration {latest.get('iteration')}", color=TEXT, fontsize=12, loc="left")
        outputs.append(_save(fig, output / "latest_evaluation.png"))

    if promotions:
        fig, ax = plt.subplots(figsize=(8, 2.8))
        _style_axis(ax)
        xs = [p.get("iteration", 0) for p in promotions]
        ys = [p.get("score", 0.0) for p in promotions]
        ax.scatter(xs, ys, color=GREEN, s=45, zorder=3)
        ax.plot(xs, ys, color=GREEN, alpha=0.5)
        ax.set_ylim(0, 1)
        ax.set_title("Candidate promotions", color=TEXT, fontsize=13, loc="left")
        ax.set_xlabel("iteration", color=DIM)
        ax.set_ylabel("gate score", color=DIM)
        outputs.append(_save(fig, output / "promotions.png"))

    summary = {
        "experiment": experiment,
        "iterations": len(metrics),
        "latest": metrics[-1],
        "candidate_evaluations": candidate_records,
        "promotions": promotions,
        "charts": [str(path) for path in outputs],
    }
    summary_path = output / "report_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    outputs.append(summary_path)
    return outputs


def generate_before_after_chart(config: dict, output_dir: Optional[str | Path] = None) -> Optional[Path]:
    path = resolve_path(config, "results") / "ui_comparisons.json"
    history = read_json(path, []) or []
    if not history:
        return None
    latest = history[-1]
    metadata = latest.get("metadata", {}) or {}
    experiment = metadata.get("trained_experiment") or metadata.get("initial_experiment") or "default"
    output = Path(output_dir) if output_dir else results_dir(config, experiment) / "plots"
    fig, ax = plt.subplots(figsize=(6, 4))
    _style_axis(ax)
    values = [latest.get("wins", 0), latest.get("losses", 0), latest.get("draws", 0)]
    bars = ax.bar(["initial wins", "trained wins", "draws"], values, color=[RED, GOLD, DIM])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, str(value), ha="center", color=TEXT)
    ax.set_title("Initial vs trained (alternating colours)", color=TEXT, fontsize=12, loc="left")
    return _save(fig, output / "before_after.png")
