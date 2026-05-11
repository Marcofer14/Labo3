"""Train an approximate tabular CFR policy with exact offline Showdown rollouts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cfr.model import CFRModel
from src.cfr.offline import (
    aggregate_common,
    append_jsonl,
    stats_to_stage,
    summarize_opponents,
    train_cfr_iteration,
)
from src.cfr.neural import CFRNeuralPrior
from src.metrics.common import add_report_args, write_report_payload


def opponent_for_iteration(args: argparse.Namespace, iteration: int) -> str:
    if args.opponent_cycle:
        cycle = [item.strip() for item in args.opponent_cycle.split(",") if item.strip()]
        if cycle:
            return cycle[(iteration - 1) % len(cycle)]
    return args.opponent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train approximate tabular CFR for VGC")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--opponent", choices=["random", "greedy", "self"], default="random")
    parser.add_argument("--opponent-cycle", type=str, default="")
    parser.add_argument("--format", type=str, default="gen9vgc2026regi")
    parser.add_argument("--team", type=str, default="team.txt")
    parser.add_argument("--simulator-url", type=str, default=os.environ.get("SHOWDOWN_SIMULATOR_URL", ""))
    parser.add_argument("--simulator-timeout", type=float, default=180.0)
    parser.add_argument("--simulator-max-choices", type=int, default=8)
    parser.add_argument(
        "--simulator-opponent-policy",
        choices=["minimax", "mean", "robust"],
        default="robust",
    )
    parser.add_argument("--simulator-robust-worst-weight", type=float, default=0.35)
    parser.add_argument("--cfr-depth", type=int, default=2)
    parser.add_argument("--max-actions", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-neural-prior", action="store_true")
    parser.add_argument("--neural-checkpoint", type=Path, default=None)
    parser.add_argument("--neural-epochs", type=int, default=2)
    parser.add_argument("--neural-batch-size", type=int, default=64)
    parser.add_argument("--neural-lr", type=float, default=3e-4)
    parser.add_argument("--neural-prior-weight", type=float, default=0.45)
    parser.add_argument("--neural-target-temperature", type=float, default=0.35)
    parser.add_argument("--neural-replay-size", type=int, default=20000)
    parser.add_argument("--neural-hidden-size", type=int, default=256)
    parser.add_argument("--neural-embedding-size", type=int, default=192)
    parser.add_argument("--neural-layers", type=int, default=3)
    parser.add_argument("--neural-dropout", type=float, default=0.05)
    parser.add_argument("--neural-policy-smoothing", type=float, default=0.02)
    parser.add_argument("--min-average-visits", type=int, default=3)
    parser.add_argument("--offline-max-turns", type=int, default=120)
    parser.add_argument("--offline-team-preview", choices=["default", "random"], default="random")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--progress-games", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint CFR JSON inicial.")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/cfr_tabular"))
    parser.add_argument("--rollout-path", type=Path, default=Path("data/cfr/rollouts.jsonl"))
    parser.add_argument(
        "--training-log-path",
        type=Path,
        default=Path("logs/cfr_train_events.jsonl"),
        help="JSONL con eventos y metricas del entrenamiento CFR.",
    )
    add_report_args(parser)
    return parser.parse_args()


def log_event(args: argparse.Namespace, event: str, **payload: Any) -> None:
    append_jsonl(
        args.training_log_path,
        {
            "event": event,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **payload,
        },
    )


def stage_loss_row(stage: dict[str, Any]) -> dict[str, Any]:
    neural = stage.get("neural") or {}
    model = stage.get("model") or {}
    return {
        "iteration": int(stage.get("stage") or 0),
        "epoch": 1,
        "loss": stage.get("avg_positive_regret"),
        "policy_loss": stage.get("avg_strategy_entropy"),
        "value_loss": stage.get("avg_reward"),
        "mcts_ce": None,
        "entropy": stage.get("avg_strategy_entropy"),
        "neural_loss": neural.get("loss"),
        "neural_policy_loss": neural.get("policy_loss"),
        "neural_value_loss": neural.get("value_loss"),
        "information_sets": model.get("information_sets"),
        "visited_information_sets": model.get("visited_information_sets"),
        "total_visits": model.get("total_visits"),
        "max_positive_regret": model.get("max_positive_regret"),
    }


def build_report(args: argparse.Namespace, stages: list[dict[str, Any]], checkpoint_path: Path) -> dict[str, Any]:
    common = aggregate_common(
        stages,
        checkpoint_path=checkpoint_path,
        log_path=args.training_log_path,
        rollout_path=args.rollout_path,
    )
    if not args.no_neural_prior:
        common["model_family"] = "cfr_tabular_neural"
        common["algorithm"] = "cfr_tabular_neural_prior"
    reward_breakdown: dict[str, float] = {}
    for stage in stages:
        for key, value in (stage.get("reward_breakdown") or {}).items():
            reward_breakdown[key] = reward_breakdown.get(key, 0.0) + float(value or 0.0)
    return {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "stages": stages,
        "training_config": {
            "algorithm": "cfr_tabular_neural_prior" if not args.no_neural_prior else "cfr_tabular_approx",
            "iterations": args.iterations,
            "games": args.games,
            "opponent": args.opponent,
            "opponent_cycle": args.opponent_cycle,
            "format": args.format,
            "team": str(args.team),
            "cfr_depth": args.cfr_depth,
            "max_actions": args.max_actions,
            "neural_prior": not args.no_neural_prior,
            "neural_checkpoint": str(args.neural_checkpoint or (args.output_dir / "prior.pt")),
            "neural_epochs": args.neural_epochs,
            "neural_batch_size": args.neural_batch_size,
            "neural_lr": args.neural_lr,
            "neural_prior_weight": args.neural_prior_weight,
            "neural_target_temperature": args.neural_target_temperature,
            "neural_replay_size": args.neural_replay_size,
            "neural_hidden_size": args.neural_hidden_size,
            "neural_embedding_size": args.neural_embedding_size,
            "neural_layers": args.neural_layers,
            "neural_dropout": args.neural_dropout,
            "neural_policy_smoothing": args.neural_policy_smoothing,
            "min_average_visits": args.min_average_visits,
            "simulator_url": args.simulator_url,
            "simulator_timeout": args.simulator_timeout,
            "simulator_max_choices": args.simulator_max_choices,
            "simulator_opponent_policy": args.simulator_opponent_policy,
            "simulator_robust_worst_weight": args.simulator_robust_worst_weight,
            "offline_max_turns": args.offline_max_turns,
            "output_dir": str(args.output_dir),
            "rollout_path": str(args.rollout_path),
            "training_log_path": str(args.training_log_path),
        },
        "extra_meta": {
            "title": "CFR tabular + red prior" if not args.no_neural_prior else "CFR tabular aproximado",
            "status": "finished",
            "loss_names": {
                "final_loss": "avg positive regret",
                "final_policy_loss": "avg strategy entropy",
                "neural_loss": "supervised utility-prior loss",
            },
        },
        "common_metrics": common,
        "reward_breakdown": reward_breakdown,
        "league": summarize_opponents(stages),
        "loss_history": [stage_loss_row(stage) for stage in stages],
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best.json"
    if args.checkpoint and args.checkpoint.exists():
        model = CFRModel.load(args.checkpoint)
        print(f"Loaded CFR checkpoint: {args.checkpoint}")
    elif checkpoint_path.exists():
        model = CFRModel.load(checkpoint_path)
        print(f"Loaded CFR checkpoint: {checkpoint_path}")
    else:
        model = CFRModel(metadata={"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        print("Initialized new CFR tabular model.")

    neural_path = args.neural_checkpoint or (args.output_dir / "prior.pt")
    neural_prior: CFRNeuralPrior | None = None
    neural_replay: list[dict[str, Any]] = []
    if not args.no_neural_prior:
        neural_prior = CFRNeuralPrior(
            checkpoint_path=neural_path if neural_path.exists() else None,
            device=args.device,
            lr=args.neural_lr,
            hidden_size=args.neural_hidden_size,
            embedding_size=args.neural_embedding_size,
            layers=args.neural_layers,
            dropout=args.neural_dropout,
            policy_smoothing=args.neural_policy_smoothing,
        )
        print(
            f"Neural prior: checkpoint={neural_path} "
            f"ready={neural_prior.ready} device={args.device}"
        )

    log_event(
        args,
        "run_start",
        iterations=args.iterations,
        games=args.games,
        opponent=args.opponent,
        opponent_cycle=args.opponent_cycle,
        cfr_depth=args.cfr_depth,
        max_actions=args.max_actions,
        neural_prior=not args.no_neural_prior,
        neural_checkpoint=str(neural_path),
        neural_prior_weight=args.neural_prior_weight,
        neural_hidden_size=args.neural_hidden_size,
        neural_embedding_size=args.neural_embedding_size,
        neural_layers=args.neural_layers,
        neural_dropout=args.neural_dropout,
        neural_policy_smoothing=args.neural_policy_smoothing,
        min_average_visits=args.min_average_visits,
        output_dir=str(args.output_dir),
        checkpoint_path=str(checkpoint_path),
    )

    stages: list[dict[str, Any]] = []
    cumulative_decisions = 0
    try:
        for iteration in range(1, args.iterations + 1):
            opponent = opponent_for_iteration(args, iteration)
            print(f"\n=== CFR iteration {iteration}/{args.iterations} ===")
            print(f"Curriculum opponent: {opponent}")
            log_event(args, "iteration_start", iteration=iteration, opponent=opponent)
            iteration_examples: list[dict[str, Any]] = []
            stats = train_cfr_iteration(
                args,
                model,
                iteration=iteration,
                opponent=opponent,
                neural_prior=neural_prior,
                neural_examples=iteration_examples,
            )
            model.metadata.update(
                {
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "iterations_completed": iteration,
                    "format": args.format,
                    "neural_prior_checkpoint": str(neural_path) if neural_prior is not None else "",
                }
            )
            model.save(checkpoint_path)
            stage = stats_to_stage(stats, cumulative_decisions, model.stats())
            if neural_prior is not None and iteration_examples:
                neural_replay.extend(iteration_examples)
                if args.neural_replay_size > 0 and len(neural_replay) > args.neural_replay_size:
                    neural_replay = neural_replay[-int(args.neural_replay_size) :]
                neural_metrics = neural_prior.fit(
                    neural_replay,
                    epochs=args.neural_epochs,
                    batch_size=args.neural_batch_size,
                )
                neural_prior.save(neural_path)
                stage["neural"] = {
                    "checkpoint": str(neural_path),
                    "architecture": getattr(neural_prior.model, "model_type", "alphazero_ranker"),
                    "model_config": neural_prior.model.config(),
                    "examples_iteration": len(iteration_examples),
                    "examples_replay": len(neural_replay),
                    **neural_metrics,
                }
            stages.append(stage)
            cumulative_decisions += stats.decisions
            log_event(args, "iteration_finished", **stage)
            print(
                f"CFR stats: games={stats.games} wins={stats.wins} losses={stats.losses} "
                f"draws={stats.draws} decisions={stats.decisions} "
                f"trunc={stats.truncated_games} score_wr={stats.score_adjusted_win_rate:.2f} "
                f"avg_regret={stats.avg_positive_regret:.4f} entropy={stats.avg_entropy:.4f}"
            )
            if stage.get("neural"):
                neural = stage["neural"]
                print(
                    f"Neural prior: loss={neural['loss']:.4f} "
                    f"policy={neural['policy_loss']:.4f} value={neural['value_loss']:.4f} "
                    f"examples={neural['examples_replay']}"
                )
            print(f"Saved CFR checkpoint to {checkpoint_path}")
    except Exception as exc:
        log_event(args, "run_failed", error_type=type(exc).__name__, error=str(exc))
        if not args.no_metrics_report and stages:
            report = build_report(args, stages, checkpoint_path)
            report["extra_meta"]["status"] = "failed"
            out = write_report_payload(report, output_root=args.metrics_report_dir, prefix="cfr")
            print(f"\n  ✓ Reporte CFR parcial generado en: {out}")
        raise

    log_event(args, "run_finished")
    if not args.no_metrics_report:
        report = build_report(args, stages, checkpoint_path)
        out = write_report_payload(report, output_root=args.metrics_report_dir, prefix="cfr")
        print(f"\n  ✓ Reporte CFR generado en: {out}")
        print(f"    · {out / 'report.html'}")
        print(f"    · {out / 'report.json'}")
        print(f"    · {out / 'common_metrics.json'}")
        print(f"    · {out / 'league_stats.json'}")
        print(f"    · {out / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
