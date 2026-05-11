"""Evaluate a tabular CFR checkpoint in exact offline Showdown battles."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from login import load_team, should_use_team
from src.cfr.model import CFRModel
from src.cfr.neural import CFRNeuralPrior, mix_strategies
from src.cfr.offline import (
    CFRShowdownClient,
    aggregate_common,
    baseline_strategy,
    outcome_for_side,
    sample_action,
    score_for_side,
    team_order,
)
from src.cfr.state_abstraction import offline_state_key
from src.format_resolver import resolve_format
from src.metrics.common import write_report_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CFR offline")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--p2", choices=["random", "greedy", "self"], default="random")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--format", type=str, default="gen9vgc2026regi")
    parser.add_argument("--team", type=str, default="team.txt")
    parser.add_argument("--simulator-url", type=str, default="http://showdown-sim:9001")
    parser.add_argument("--simulator-timeout", type=float, default=180.0)
    parser.add_argument("--max-actions", type=int, default=8)
    parser.add_argument("--offline-max-turns", type=int, default=120)
    parser.add_argument("--offline-team-preview", choices=["default", "random"], default="random")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--neural-checkpoint", type=Path, default=None)
    parser.add_argument("--neural-weight", type=float, default=0.70)
    parser.add_argument("--min-average-visits", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--report-dir", type=Path, default=None)
    return parser.parse_args()


def select_cfr_action(
    model: CFRModel,
    snapshot: dict[str, Any],
    side: str,
    actions: list[str],
    rng: random.Random,
    temperature: float,
    *,
    neural_prior: CFRNeuralPrior | None = None,
    neural_weight: float = 0.70,
    min_average_visits: int = 3,
) -> str:
    if not actions:
        return ""
    key = offline_state_key(snapshot, side)
    node = model.nodes.get(key)
    strategy: dict[str, float] = {}
    if node is not None and node.visits > 0:
        strategy = model.strategy(
            key,
            actions,
            average=node.visits >= max(1, int(min_average_visits)),
        )
    if neural_prior is not None and neural_prior.ready:
        prior_strategy, _ = neural_prior.predict_offline(snapshot, side, actions)
        if strategy:
            weight = float(neural_weight)
            if node is not None and node.visits < max(1, int(min_average_visits)):
                weight = max(weight, 1.0 - node.visits / max(1, int(min_average_visits)))
            strategy = mix_strategies(actions, strategy, prior_strategy, weight)
        else:
            strategy = prior_strategy
    if not strategy:
        return actions[0]
    if temperature <= 0:
        return max(actions, key=lambda action: strategy.get(action, 0.0))
    threshold = rng.random()
    cumulative = 0.0
    selected = actions[-1]
    for action in actions:
        cumulative += float(strategy.get(action, 0.0))
        if threshold <= cumulative:
            selected = action
            break
    return selected


def main() -> int:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    battle_format = resolve_format(args.format)
    if not should_use_team(battle_format):
        raise ValueError("Offline CFR evaluation requires a fixed team format")

    model = CFRModel.load(args.checkpoint)
    neural_prior = (
        CFRNeuralPrior(checkpoint_path=args.neural_checkpoint, device=args.device)
        if args.neural_checkpoint and args.neural_checkpoint.exists()
        else None
    )
    client = CFRShowdownClient(args.simulator_url, timeout=args.simulator_timeout)
    team_text = load_team(args.team)
    wins = losses = draws = decisions = turns_total = 0
    truncated = score_adjusted_wins = score_adjusted_losses = 0
    score_total = 0.0

    print("=" * 55)
    print("  VGC Bot - CFR Offline Eval")
    print("=" * 55)
    print(f"  Formato: {battle_format}")
    print(f"  Checkpoint: {args.checkpoint}")
    if neural_prior is not None:
        print(f"  Neural prior: {args.neural_checkpoint}")
    print(f"  Rival: {args.p2.upper()}")
    print(f"  Partidas: {args.n}")
    print("=" * 55)

    started = time.time()
    for game_index in range(args.n):
        rng = random.Random(args.seed + game_index)
        seed = [
            rng.randrange(1, 0xFFFFFFFF),
            rng.randrange(1, 0xFFFFFFFF),
            rng.randrange(1, 0xFFFFFFFF),
            rng.randrange(1, 0xFFFFFFFF),
        ]
        snapshot = client.start(
            battle_format=battle_format,
            team_text=team_text,
            seed=seed,
            team_choice_p1=team_order(rng, team_text, args.offline_team_preview),
            team_choice_p2=team_order(rng, team_text, args.offline_team_preview),
            max_choices=args.max_actions,
        )
        for _ in range(max(1, int(args.offline_max_turns))):
            if snapshot.get("ended"):
                break
            legal = snapshot.get("legal") or {}
            p1_actions = list(legal.get("p1") or [])[: args.max_actions]
            p2_actions = list(legal.get("p2") or [])[: args.max_actions]
            choices: dict[str, str] = {}
            if p1_actions:
                choices["p1"] = select_cfr_action(
                    model,
                    snapshot,
                    "p1",
                    p1_actions,
                    rng,
                    args.temperature,
                    neural_prior=neural_prior,
                    neural_weight=args.neural_weight,
                    min_average_visits=args.min_average_visits,
                )
            if p2_actions:
                if args.p2 == "self":
                    choices["p2"] = select_cfr_action(
                        model,
                        snapshot,
                        "p2",
                        p2_actions,
                        rng,
                        args.temperature,
                        neural_prior=neural_prior,
                        neural_weight=args.neural_weight,
                        min_average_visits=args.min_average_visits,
                    )
                else:
                    choices["p2"] = sample_action(baseline_strategy(args.p2, p2_actions), rng)
            if not choices:
                break
            snapshot = client.choose(state=snapshot["state"], choices=choices, max_choices=args.max_actions)
            decisions += 1
        reward = outcome_for_side(snapshot, "p1")
        score = score_for_side(snapshot, "p1")
        was_truncated = not bool(snapshot.get("ended"))
        if reward == 0.0 and was_truncated:
            reward = 1.0 if score >= 0.0 else -1.0
        score_total += score
        turns_total += int(snapshot.get("turn") or 0)
        if was_truncated:
            truncated += 1
        if reward > 0:
            wins += 1
            score_adjusted_wins += 1
            result = "VICTORIA*" if was_truncated else "VICTORIA"
        elif reward < 0:
            losses += 1
            score_adjusted_losses += 1
            result = "DERROTA*" if was_truncated else "DERROTA"
        else:
            draws += 1
            result = "EMPATE"
        print(f"  Final: offline-cfr-{game_index + 1} -> {result} en {snapshot.get('turn', 0)} turnos.")

    games = wins + losses + draws
    elapsed = time.time() - started
    print("\n" + "=" * 55)
    print("  RESULTADOS")
    print("=" * 55)
    print(f"  CFR victorias: {wins} / {games}")
    print(f"  Rival victorias: {losses} / {games}")
    print(f"  Empates: {draws} / {games}")
    print(f"  Truncadas: {truncated} / {games}")
    print(f"  Win rate ajustado por score: {score_adjusted_wins / games if games else 0.0:.3f}")
    print(f"  Tiempo: {elapsed:.1f}s")
    print("=" * 55)

    if args.report_dir:
        stage = {
            "stage": 1,
            "started": 0,
            "ended": decisions,
            "duration": decisions,
            "transition_reason": "evaluation_finished",
            "n_episodes": games,
            "win_rate": wins / games if games else 0.0,
            "score_adjusted_win_rate": score_adjusted_wins / games if games else 0.0,
            "avg_reward": (wins - losses) / games if games else 0.0,
            "avg_score": score_total / games if games else 0.0,
            "avg_episode_length": turns_total / games if games else 0.0,
            "final_loss": None,
            "final_policy_loss": None,
            "final_value_loss": None,
            "reward_breakdown": {"a.win": wins * 15.0, "a.loss": losses * -15.0},
            "n_updates": 0,
            "opponent": args.p2,
            "decisions": decisions,
            "truncated_games": truncated,
            "score_adjusted_wins": score_adjusted_wins,
            "score_adjusted_losses": score_adjusted_losses,
            "wall_time_sec": elapsed,
            "model": model.stats(),
        }
        common = aggregate_common(
            [stage],
            checkpoint_path=args.checkpoint,
            log_path=Path(""),
            rollout_path=Path(""),
        )
        if neural_prior is not None:
            common["model_family"] = "cfr_tabular_neural"
            common["algorithm"] = "cfr_tabular_neural_prior_eval"
        report = {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "stages": [stage],
            "training_config": {
                "algorithm": "cfr_tabular_neural_prior_eval" if neural_prior is not None else "cfr_tabular_approx_eval",
                "checkpoint": str(args.checkpoint),
                "neural_checkpoint": str(args.neural_checkpoint or ""),
                "n": args.n,
                "p2": args.p2,
                "format": args.format,
                "max_actions": args.max_actions,
                "offline_max_turns": args.offline_max_turns,
            },
            "extra_meta": {
                "title": "CFR tabular + red prior - evaluacion"
                if neural_prior is not None
                else "CFR tabular aproximado - evaluacion",
                "status": "finished",
            },
            "common_metrics": common,
            "reward_breakdown": stage["reward_breakdown"],
            "league": [],
        }
        out = write_report_payload(report, output_dir=args.report_dir, prefix="cfr_eval")
        print(f"  Reporte: {out / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
