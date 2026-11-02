"""AlphaZero-style iterative self-play training pipeline."""
from __future__ import annotations

import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from evaluation.arena import Arena, should_promote
from evaluation.metrics import EvaluationResult, save_evaluation_result
from model.checkpoint import (
    create_initial_checkpoint,
    load_checkpoint,
    load_model,
    safe_promote_best,
    save_checkpoint,
)
from model.network import AlphaZeroNetwork
from selfplay.generator import SelfPlayGenerator
from training.losses import policy_value_loss
from training.replay_buffer import ReplayBuffer
from utils.config import PROJECT_ROOT, log_dir, resolve_device, resolve_path, results_dir, training_dir
from utils.logging import (
    JsonlLogger,
    TrainingControl,
    TrainingStopped,
    read_json,
    read_jsonl,
    setup_logger,
    write_json,
)
from utils.reproducibility import seed_everything


class Trainer:
    def __init__(
        self,
        config: dict,
        experiment: Optional[str] = None,
        resume: Optional[str | Path] = None,
        device: Optional[str] = None,
        control_file: Optional[str | Path] = None,
        iterations: Optional[int] = None,
    ) -> None:
        self.config = config
        self.experiment = experiment or config.get("experiment", {}).get("name", "default")
        self.device = resolve_device(device or config.get("_device"))
        self.seed = int(config.get("experiment", {}).get("seed", 42))
        self.iterations = int(iterations if iterations is not None else config.get("training", {}).get("iterations", 50))
        seed_everything(self.seed, deterministic=bool(config.get("experiment", {}).get("deterministic", True)))

        self.checkpoint_dir = training_dir(config, self.experiment)
        self.logs_dir = log_dir(config, self.experiment)
        self.results_dir = results_dir(config, self.experiment)
        self.data_dir = self.results_dir / "selfplay"
        for directory in (self.checkpoint_dir, self.logs_dir, self.results_dir, self.data_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.initial_path = self.checkpoint_dir / "initial.pt"
        self.latest_path = self.checkpoint_dir / "latest.pt"
        self.best_path = self.checkpoint_dir / "best.pt"
        self.metrics_path = self.logs_dir / "metrics.jsonl"
        self.eval_history_path = self.results_dir / "evaluations.json"
        self.status_path = self.logs_dir / "status.json"
        self.control = TrainingControl(control_file)
        self.logger = setup_logger(self.logs_dir / "train.log")
        self.metrics_logger = JsonlLogger(self.metrics_path)
        self.rng = np.random.default_rng(self.seed)

        self.iteration = 0
        self.eval_history: List[Dict[str, Any]] = read_json(self.eval_history_path, []) or []
        self.selfplay_history: List[Dict[str, Any]] = []
        self.promotions: List[Dict[str, Any]] = []
        self._tb_writer = None
        self._setup_tensorboard()

        model_config = config.get("model", {})
        if self.initial_path.exists():
            # Preserve the original random checkpoint across all resumes.
            self.initial_model, _ = load_model(self.initial_path, device=self.device)
        else:
            self.initial_model, _ = create_initial_checkpoint(
                self.initial_path,
                {"model": model_config},
                seed=self.seed,
            )
            self.initial_model.to(self.device)

        self.best_model = self._load_best_or_initial()
        self.best_model.eval()
        self.candidate = copy.deepcopy(self.best_model).to(self.device)
        self.optimizer = self._make_optimizer(self.candidate)
        self.scheduler = self._make_scheduler(self.optimizer)

        buffer_cfg = config.get("training", {})
        self.buffer = ReplayBuffer(
            capacity=int(buffer_cfg.get("replay_capacity", 50_000)),
            observation_shape=(2, 6, 7),
            policy_size=7,
        )
        buffer_path = self.checkpoint_dir / "replay_buffer.npz"
        if resume:
            self._resume_from(resume, buffer_path)
        else:
            save_checkpoint(
                self.best_path,
                self.best_model,
                iteration=0,
                config=self.config,
                model_type="best",
                extra={"note": "Initial random model is the starting best"},
            )
            save_checkpoint(
                self.latest_path,
                self.best_model,
                iteration=0,
                config=self.config,
                model_type="latest",
            )
            self._load_buffer_if_available(buffer_path)

        self.logger.info(
            "Trainer ready | experiment=%s | iteration=%d/%d | device=%s | params=%s",
            self.experiment,
            self.iteration,
            self.iterations,
            self.device,
            sum(p.numel() for p in self.best_model.parameters()),
        )

    def _setup_tensorboard(self) -> None:
        if not bool(self.config.get("tensorboard", {}).get("enabled", True)):
            return
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = self.logs_dir / "tensorboard"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=str(tb_dir))
        except Exception as exc:  # pragma: no cover - tensorboard optional at runtime
            self.logger.warning("TensorBoard disabled: %s", exc)
            self._tb_writer = None

    def _make_optimizer(self, model: AlphaZeroNetwork) -> torch.optim.Optimizer:
        cfg = self.config.get("training", {})
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.get("learning_rate", 1e-3)),
            weight_decay=float(cfg.get("weight_decay", 1e-4)),
        )

    def _make_scheduler(self, optimizer: torch.optim.Optimizer):
        cfg = self.config.get("training", {})
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.get("lr_step_size", 999)),
            gamma=float(cfg.get("lr_gamma", 0.5)),
        )

    def _load_best_or_initial(self) -> AlphaZeroNetwork:
        if self.best_path.exists():
            try:
                model, _ = load_model(self.best_path, device=self.device)
                return model
            except Exception as exc:
                self.logger.warning("Could not load best.pt (%s); using initial model", exc)
        model = copy.deepcopy(self.initial_model).to(self.device)
        return model

    def _load_buffer_if_available(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            capacity = int(self.config.get("training", {}).get("replay_capacity", 50_000))
            self.buffer = ReplayBuffer.load(path, capacity=capacity)
            self.logger.info("Loaded replay buffer with %d examples", len(self.buffer))
        except Exception as exc:
            self.logger.warning("Could not load replay buffer: %s", exc)

    def _resume_from(self, resume: str | Path, buffer_path: Path) -> None:
        path = Path(resume)
        if not path.exists():
            candidate_path = self.checkpoint_dir / path
            if candidate_path.exists():
                path = candidate_path
        model, payload = load_checkpoint(path, device=self.device)
        self.candidate = model
        self.best_model = self._load_best_or_initial()
        self.iteration = int(payload.get("iteration", 0))
        optimizer_state = payload.get("optimizer_state")
        if optimizer_state:
            try:
                self.optimizer.load_state_dict(optimizer_state)
            except Exception as exc:
                self.logger.warning("Could not restore optimizer state: %s", exc)
        scheduler_state = payload.get("scheduler_state")
        if scheduler_state:
            try:
                self.scheduler.load_state_dict(scheduler_state)
            except Exception as exc:
                self.logger.warning("Could not restore scheduler state: %s", exc)
        self.eval_history = read_json(self.eval_history_path, []) or payload.get("extra", {}).get("evaluations", [])
        self._load_buffer_if_available(buffer_path)
        self.logger.info("Resumed from %s at iteration %d", path, self.iteration)

    def _set_status(self, status: str, **extra: Any) -> None:
        write_json(
            self.status_path,
            {
                "status": status,
                "experiment": self.experiment,
                "iteration": self.iteration,
                "total_iterations": self.iterations,
                "pid": os.getpid(),
                "buffer_size": len(self.buffer),
                "best_checkpoint": str(self.best_path),
                "updated": time.time(),
                **extra,
            },
        )

    def _check_control(self) -> None:
        state = self.control.state()
        if state["paused"]:
            self._set_status("paused")
        self.control.wait_if_paused()
        if self.control.state()["stop"]:
            raise TrainingStopped("Training stop requested.")
        self._set_status("running")

    def _selfplay_progress(self, completed: int, total: int, record) -> None:
        self._check_control()
        if completed == total or completed % max(1, total // 5) == 0:
            self.logger.info(
                "Self-play game %d/%d | moves=%d winner=%s",
                completed,
                total,
                record.length,
                record.winner or "draw",
            )

    def _generate_selfplay(self) -> tuple[List, Dict[str, Any]]:
        cfg = self.config.get("selfplay", {})
        mcts_cfg = self.config.get("mcts", {})
        num_games = int(cfg.get("games_per_iteration", 50))
        workers = int(cfg.get("workers", 1))
        self.logger.info("Iteration %d: generating %d self-play games", self.iteration + 1, num_games)
        generator = SelfPlayGenerator(
            self.best_model,
            self.config,
            device=self.device,
            simulations=int(mcts_cfg.get("simulations", 50)),
        )
        start = time.perf_counter()
        records = generator.generate_games(
            num_games,
            base_seed=self.seed + 10_000 * (self.iteration + 1),
            workers=workers,
            progress_callback=self._selfplay_progress,
        )
        elapsed = time.perf_counter() - start
        if cfg.get("save_examples", True):
            from selfplay.generator import save_game_records

            save_game_records(records, self.data_dir / f"iteration_{self.iteration + 1:04d}.npz")

        lengths = [r.length for r in records]
        p1_wins = sum(1 for r in records if r.winner == 1)
        p2_wins = sum(1 for r in records if r.winner == 2)
        draws = sum(1 for r in records if r.winner == 0)
        examples = sum(len(r.examples) for r in records)
        stats = {
            "games": len(records),
            "examples": examples,
            "avg_game_length": float(np.mean(lengths)) if lengths else 0.0,
            "selfplay_p1_wins": p1_wins,
            "selfplay_p2_wins": p2_wins,
            "selfplay_draws": draws,
            "selfplay_seconds": elapsed,
        }
        for record in records:
            observations, policies, values = record.arrays()
            self.buffer.add(observations, policies, values)
        self.selfplay_history.append({"iteration": self.iteration + 1, **stats})
        self.logger.info(
            "Self-play complete: %d games, %d examples, avg length %.1f, %.1fs",
            len(records),
            examples,
            stats["avg_game_length"],
            elapsed,
        )
        return records, stats

    def _train_network(self) -> Dict[str, float]:
        cfg = self.config.get("training", {})
        epochs = int(cfg.get("epochs", 5))
        batch_size = int(cfg.get("batch_size", 128))
        max_batches = int(cfg.get("max_batches_per_epoch", 64))
        l2_coef = float(cfg.get("l2_coefficient", 1e-4))
        clip = float(cfg.get("gradient_clip", 1.0))
        min_buffer = int(cfg.get("min_buffer_size", 32))

        if len(self.buffer) < max(2, min_buffer):
            self.logger.warning("Replay buffer too small (%d), skipping training", len(self.buffer))
            return {
                "policy_loss": float("nan"),
                "value_loss": float("nan"),
                "total_loss": float("nan"),
                "l2_loss": float("nan"),
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                "train_batches": 0.0,
                "training_seconds": 0.0,
            }

        self.candidate = copy.deepcopy(self.best_model).to(self.device)
        self.optimizer = self._make_optimizer(self.candidate)
        self.scheduler = self._make_scheduler(self.optimizer)
        # Do not leak BatchNorm running statistics between train/eval phases.
        self.candidate.train()
        start = time.perf_counter()
        totals = {"loss": 0.0, "policy": 0.0, "value": 0.0, "l2": 0.0}
        batches = 0
        for epoch in range(epochs):
            self._check_control()
            batches_this_epoch = min(max_batches, math.ceil(len(self.buffer) / batch_size))
            for _ in range(batches_this_epoch):
                observations, policies, values = self.buffer.sample(batch_size, rng=self.rng)
                x = torch.from_numpy(observations).to(self.device, dtype=torch.float32)
                pi = torch.from_numpy(policies).to(self.device, dtype=torch.float32)
                z = torch.from_numpy(values).to(self.device, dtype=torch.float32)
                policy, value = self.candidate(x)
                total, policy_loss, value_loss, l2_loss = policy_value_loss(
                    policy,
                    value,
                    pi,
                    z,
                    parameters=self.candidate.parameters(),
                    l2_coefficient=l2_coef,
                )
                self.optimizer.zero_grad(set_to_none=True)
                total.backward()
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.candidate.parameters(), clip)
                self.optimizer.step()
                totals["loss"] += float(total.item())
                totals["policy"] += float(policy_loss.item())
                totals["value"] += float(value_loss.item())
                totals["l2"] += float(l2_loss.item())
                batches += 1
            self.scheduler.step()
            self.logger.info(
                "Epoch %d/%d complete | avg loss %.4f | lr %.6g",
                epoch + 1,
                epochs,
                totals["loss"] / max(1, batches),
                self.optimizer.param_groups[0]["lr"],
            )
        elapsed = time.perf_counter() - start
        self.candidate.eval()
        return {
            "policy_loss": totals["policy"] / max(1, batches),
            "value_loss": totals["value"] / max(1, batches),
            "total_loss": totals["loss"] / max(1, batches),
            "l2_loss": totals["l2"] / max(1, batches),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "train_batches": float(batches),
            "training_seconds": elapsed,
        }

    def _run_evaluations(self) -> Dict[str, Any]:
        cfg = self.config.get("evaluation", {})
        simulations = int(cfg.get("simulations", self.config.get("mcts", {}).get("simulations", 50)))
        candidate_games = int(cfg.get("candidate_games", 20))
        random_games = int(cfg.get("random_games", 20))
        tactical_games = int(cfg.get("tactical_games", 0))
        threshold = float(cfg.get("promotion_threshold", 0.55))
        arena = Arena(self.config)

        self.logger.info("Evaluating candidate against current best (%d games)", candidate_games)
        vs_best = arena.evaluate_models(
            self.candidate,
            self.best_model,
            games=candidate_games,
            device=self.device,
            simulations=simulations,
            name_a=f"candidate_{self.iteration + 1}",
            name_b="best",
            seed=self.seed + 7000 + self.iteration,
            iteration=self.iteration + 1,
        )
        promoted = should_promote(vs_best, threshold)
        record = vs_best.as_dict()
        record.update({"type": "candidate_vs_best", "promoted": promoted})
        self.eval_history.append(record)
        save_evaluation_result(
            vs_best,
            self.eval_history_path,
            record_type="candidate_vs_best",
            extra={"promoted": promoted},
        )
        self.logger.info("Candidate vs best: %s | promoted=%s", vs_best.summary(), promoted)

        if promoted:
            self.best_model = copy.deepcopy(self.candidate).to(self.device)
            self.best_model.eval()
            safe_promote_best(
                self.best_path,
                self.best_model,
                iteration=self.iteration + 1,
                config=self.config,
                extra={"evaluations": self.eval_history},
            )
            self.promotions.append(
                {
                    "iteration": self.iteration + 1,
                    "score": vs_best.score,
                    "wins": vs_best.wins,
                    "losses": vs_best.losses,
                    "draws": vs_best.draws,
                }
            )
            self.logger.info("Candidate promoted to best at iteration %d", self.iteration + 1)

        vs_random = None
        vs_tactical = None
        if random_games > 0:
            self.logger.info("Evaluating best against random agent (%d games)", random_games)
            vs_random = arena.evaluate_against_random(
                self.best_model,
                games=random_games,
                device=self.device,
                simulations=simulations,
                model_name="best",
                seed=self.seed + 9000 + self.iteration,
                iteration=self.iteration + 1,
            )
            record = vs_random.as_dict()
            record["type"] = "best_vs_random"
            self.eval_history.append(record)
            save_evaluation_result(vs_random, self.eval_history_path, record_type="best_vs_random")
            self.logger.info("Best vs random: %s", vs_random.summary())
        if tactical_games > 0:
            self.logger.info("Evaluating best against tactical baseline (%d games)", tactical_games)
            vs_tactical = arena.evaluate_against_tactical(
                self.best_model,
                games=tactical_games,
                device=self.device,
                simulations=simulations,
                model_name="best",
                seed=self.seed + 11000 + self.iteration,
                iteration=self.iteration + 1,
            )
            record = vs_tactical.as_dict()
            record["type"] = "best_vs_tactical"
            self.eval_history.append(record)
            save_evaluation_result(vs_tactical, self.eval_history_path, record_type="best_vs_tactical")
            self.logger.info("Best vs tactical: %s", vs_tactical.summary())

        return {
            "vs_best": vs_best,
            "promoted": promoted,
            "vs_random": vs_random,
            "vs_tactical": vs_tactical,
        }

    def _save_iteration_artifacts(self, metrics: Dict[str, Any], evaluation: Dict[str, Any]) -> None:
        iteration = self.iteration
        extra = {
            "evaluations": self.eval_history,
            "promotions": self.promotions,
            "selfplay_history": self.selfplay_history,
            "latest_metrics": metrics,
        }
        iteration_path = self.checkpoint_dir / f"iteration_{iteration:04d}.pt"
        save_checkpoint(
            iteration_path,
            self.candidate,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            iteration=iteration,
            config=self.config,
            extra=extra,
            model_type="candidate",
        )
        save_checkpoint(
            self.latest_path,
            self.candidate,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            iteration=iteration,
            config=self.config,
            extra=extra,
            model_type="latest",
        )
        if bool(self.config.get("training", {}).get("save_buffer", True)):
            self.buffer.save(self.checkpoint_dir / "replay_buffer.npz")
        write_json(self.checkpoint_dir / "evaluation_history.json", self.eval_history)
        write_json(self.checkpoint_dir / "promotions.json", self.promotions)

    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        self.metrics_logger.write(metrics)
        self.logger.info(
            "Iteration %d metrics | policy %.4f | value %.4f | total %.4f | "
            "buffer %d | vs_random %s | vs_best %s | promoted %s",
            metrics["iteration"],
            metrics.get("policy_loss", float("nan")),
            metrics.get("value_loss", float("nan")),
            metrics.get("total_loss", float("nan")),
            metrics["buffer_size"],
            metrics.get("win_rate_vs_random"),
            metrics.get("candidate_score_vs_best"),
            metrics.get("promoted"),
        )
        if self._tb_writer is not None:
            step = metrics["iteration"]
            for key in (
                "policy_loss",
                "value_loss",
                "total_loss",
                "l2_loss",
                "learning_rate",
                "buffer_size",
                "games_played",
                "avg_game_length",
                "win_rate_vs_random",
                "candidate_score_vs_best",
                "win_rate_vs_tactical",
                "selfplay_seconds",
                "training_seconds",
                "evaluation_seconds",
            ):
                value = metrics.get(key)
                if value is not None and np.isfinite(value):
                    self._tb_writer.add_scalar(f"training/{key}", value, step)
            self._tb_writer.add_scalar("training/promoted", int(bool(metrics.get("promoted"))), step)
            self._tb_writer.flush()

    def run_iteration(self) -> Dict[str, Any]:
        self._check_control()
        self._set_status("running", phase="selfplay")
        records, selfplay_stats = self._generate_selfplay()

        self._set_status("running", phase="training")
        self.logger.info("Training candidate network for %d epochs", int(self.config.get("training", {}).get("epochs", 5)))
        train_stats = self._train_network()
        if self._tb_writer is not None:
            self._tb_writer.add_scalar("training/replay_buffer_size", len(self.buffer), self.iteration + 1)

        self._check_control()
        self._set_status("running", phase="evaluation")
        eval_start = time.perf_counter()
        evaluation = self._run_evaluations()
        eval_seconds = time.perf_counter() - eval_start
        self.iteration += 1

        vs_best: EvaluationResult = evaluation["vs_best"]
        vs_random: Optional[EvaluationResult] = evaluation["vs_random"]
        vs_tactical: Optional[EvaluationResult] = evaluation["vs_tactical"]
        metrics: Dict[str, Any] = {
            "iteration": self.iteration,
            "games_played": selfplay_stats["games"],
            "examples_generated": selfplay_stats["examples"],
            "avg_game_length": selfplay_stats["avg_game_length"],
            "selfplay_p1_wins": selfplay_stats["selfplay_p1_wins"],
            "selfplay_p2_wins": selfplay_stats["selfplay_p2_wins"],
            "selfplay_draws": selfplay_stats["selfplay_draws"],
            "buffer_size": len(self.buffer),
            "policy_loss": train_stats["policy_loss"],
            "value_loss": train_stats["value_loss"],
            "total_loss": train_stats["total_loss"],
            "l2_loss": train_stats["l2_loss"],
            "learning_rate": train_stats["learning_rate"],
            "train_batches": train_stats["train_batches"],
            "selfplay_seconds": selfplay_stats["selfplay_seconds"],
            "training_seconds": train_stats["training_seconds"],
            "evaluation_seconds": eval_seconds,
            "candidate_score_vs_best": vs_best.score,
            "candidate_win_rate_vs_best": vs_best.win_rate,
            "candidate_vs_best_wins": vs_best.wins,
            "candidate_vs_best_losses": vs_best.losses,
            "candidate_vs_best_draws": vs_best.draws,
            "promoted": bool(evaluation["promoted"]),
            "win_rate_vs_random": vs_random.win_rate if vs_random else None,
            "score_vs_random": vs_random.score if vs_random else None,
            "win_rate_vs_tactical": vs_tactical.win_rate if vs_tactical else None,
            "score_vs_tactical": vs_tactical.score if vs_tactical else None,
            "best_checkpoint": str(self.best_path),
            "promotions_total": len(self.promotions),
            "device": str(self.device),
        }
        self._save_iteration_artifacts(metrics, evaluation)
        self._log_metrics(metrics)
        self._set_status("running", phase="iteration_complete", last_metrics=metrics)
        return metrics

    def train(self) -> List[Dict[str, Any]]:
        results = []
        self._set_status("running", phase="starting")
        try:
            while self.iteration < self.iterations:
                self._check_control()
                metrics = self.run_iteration()
                results.append(metrics)
            self._set_status("completed")
            self.logger.info("Training complete at iteration %d", self.iteration)
        except TrainingStopped:
            self.logger.info("Training stopped safely at iteration %d", self.iteration)
            self._set_status("stopped")
        except KeyboardInterrupt:
            self.logger.info("Training interrupted at iteration %d", self.iteration)
            self._set_status("stopped")
        except Exception as exc:
            self.logger.exception("Training failed: %s", exc)
            self._set_status("failed", error=str(exc))
            raise
        finally:
            if self._tb_writer is not None:
                self._tb_writer.close()
        return results

    def close(self) -> None:
        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None
