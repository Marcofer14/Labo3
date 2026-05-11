"""Offline tabular CFR training helpers."""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from login import load_team, should_use_team
from src.cfr.model import CFRModel, entropy, normalize_action
from src.cfr.neural import CFRNeuralPrior, mix_strategies
from src.cfr.state_abstraction import offline_state_key
from src.format_resolver import resolve_format


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


class CFRShowdownClient:
    def __init__(self, url: str | None = None, *, timeout: float = 60.0):
        self.url = (url or os.environ.get("SHOWDOWN_SIMULATOR_URL") or "").rstrip("/")
        self.timeout = timeout
        if not self.url:
            raise ValueError("SHOWDOWN_SIMULATOR_URL is required for offline CFR")

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.url}{path}",
            data=data,
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"CFR simulator HTTP {exc.code}: {detail[:1000]}") from exc
        except TimeoutError as exc:
            raise RuntimeError(f"CFR simulator timed out after {self.timeout:.1f}s on {path}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"CFR simulator request failed on {path}: {exc}") from exc
        if not result.get("ok"):
            raise RuntimeError(str(result.get("error") or "simulator returned ok=false"))
        return result

    def start(
        self,
        *,
        battle_format: str,
        team_text: str,
        seed: list[int],
        team_choice_p1: str,
        team_choice_p2: str,
        max_choices: int,
    ) -> dict[str, Any]:
        result = self._post(
            "/offline/start",
            {
                "format": battle_format,
                "team_p1": team_text,
                "team_p2": team_text,
                "seed": seed,
                "team_choice_p1": team_choice_p1,
                "team_choice_p2": team_choice_p2,
                "max_choices": max_choices,
            },
        )
        return result["battle"]

    def choose(self, *, state: Any, choices: dict[str, str], max_choices: int) -> dict[str, Any]:
        result = self._post(
            "/offline/choose",
            {"state": state, "choices": choices, "max_choices": max_choices},
        )
        return result["battle"]

    def matrix(
        self,
        *,
        state: Any,
        side: str,
        candidates: list[str],
        opponent_candidates: list[str],
        depth: int,
        max_choices: int,
        opponent_policy: str,
        robust_worst_weight: float,
    ) -> dict[str, Any]:
        return self._post(
            "/offline/matrix",
            {
                "state": state,
                "side": side,
                "candidates": candidates,
                "opponent_candidates": opponent_candidates,
                "depth": depth,
                "max_choices": max_choices,
                "opponent_policy": opponent_policy,
                "robust_worst_weight": robust_worst_weight,
            },
        )


@dataclass
class CFRIterationStats:
    iteration: int
    opponent: str
    decisions: int = 0
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    truncated_games: int = 0
    score_adjusted_wins: int = 0
    score_adjusted_losses: int = 0
    total_reward: float = 0.0
    total_score: float = 0.0
    total_turns: int = 0
    regret_updates: int = 0
    positive_regret_sum: float = 0.0
    entropy_sum: float = 0.0
    simulator_errors: int = 0
    skipped_branches: int = 0
    started_at: float = field(default_factory=time.time)
    ended_at: float = 0.0

    def finish(self) -> None:
        self.ended_at = time.time()

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.games if self.games else 0.0

    @property
    def avg_turns(self) -> float:
        return self.total_turns / self.games if self.games else 0.0

    @property
    def avg_score(self) -> float:
        return self.total_score / self.games if self.games else 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def score_adjusted_win_rate(self) -> float:
        return self.score_adjusted_wins / self.games if self.games else 0.0

    @property
    def avg_positive_regret(self) -> float:
        return self.positive_regret_sum / self.regret_updates if self.regret_updates else 0.0

    @property
    def avg_entropy(self) -> float:
        return self.entropy_sum / self.regret_updates if self.regret_updates else 0.0


def team_order(rng: random.Random, team_text: str, mode: str) -> str:
    members = [block for block in team_text.split("\n\n") if block.strip()]
    count = max(1, len(members))
    size = min(4, count)
    if mode == "random" and count >= size:
        order = rng.sample(range(1, count + 1), size)
    else:
        order = list(range(1, size + 1))
    return "team " + "".join(str(item) for item in order)


def outcome_for_side(snapshot: dict[str, Any], side: str) -> float:
    winner = snapshot.get("winner_side") or ""
    if winner == side:
        return 1.0
    if winner in {"p1", "p2"}:
        return -1.0
    return 0.0


def score_for_side(snapshot: dict[str, Any], side: str) -> float:
    try:
        return float((snapshot.get("score") or {}).get(side) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def baseline_strategy(kind: str, actions: list[str]) -> dict[str, float]:
    actions = [normalize_action(action) for action in actions]
    if not actions:
        return {}
    if kind == "random":
        probability = 1.0 / len(actions)
        return {action: probability for action in actions}
    if kind == "greedy":
        return {action: 1.0 if index == 0 else 0.0 for index, action in enumerate(actions)}
    raise ValueError(f"Unsupported CFR opponent: {kind}")


def sample_action(strategy: dict[str, float], rng: random.Random) -> str:
    if not strategy:
        return ""
    threshold = rng.random()
    cumulative = 0.0
    last = ""
    for action, probability in strategy.items():
        last = action
        cumulative += float(probability)
        if threshold <= cumulative:
            return action
    return last


def is_auto_default(actions: list[str]) -> bool:
    return len(actions) == 1 and normalize_action(actions[0]) == "default"


def strategic_actions(actions: list[str]) -> list[str]:
    return [action for action in actions if normalize_action(action) != "default"]


def utilities_against_strategy(matrix: np.ndarray, opponent_strategy: dict[str, float], opponent_actions: list[str]) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float32)
    weights = np.asarray(
        [float(opponent_strategy.get(normalize_action(action), 0.0)) for action in opponent_actions],
        dtype=np.float32,
    )
    total = float(weights.sum())
    if total <= 1e-8:
        weights = np.full(len(opponent_actions), 1.0 / max(1, len(opponent_actions)), dtype=np.float32)
    else:
        weights /= total
    return matrix @ weights


def cfr_update(
    *,
    model: CFRModel,
    client: CFRShowdownClient,
    snapshot: dict[str, Any],
    side: str,
    own_actions: list[str],
    opponent_actions: list[str],
    opponent_strategy: dict[str, float],
    depth: int,
    simulator_max_choices: int,
    simulator_opponent_policy: str,
    simulator_robust_worst_weight: float,
    neural_prior: CFRNeuralPrior | None = None,
    neural_examples: list[dict[str, Any]] | None = None,
    neural_prior_weight: float = 0.0,
    neural_target_temperature: float = 0.35,
    min_average_visits: int = 3,
) -> tuple[dict[str, float], float, float, int, int]:
    key = offline_state_key(snapshot, side)
    own_actions = [normalize_action(action) for action in own_actions]
    opponent_actions = [normalize_action(action) for action in opponent_actions]
    node = model.node(key)
    tabular_strategy = node.strategy(own_actions)
    strategy = tabular_strategy
    if neural_prior is not None and neural_prior.ready:
        prior_strategy, _ = neural_prior.predict_offline(snapshot, side, own_actions)
        if node.visits <= 0:
            prior_weight = 1.0
        elif node.visits < max(1, int(min_average_visits)):
            prior_weight = max(
                float(neural_prior_weight),
                1.0 - (node.visits / max(1, int(min_average_visits))),
            )
        else:
            prior_weight = float(neural_prior_weight)
        strategy = mix_strategies(own_actions, tabular_strategy, prior_strategy, prior_weight)
    result = client.matrix(
        state=snapshot["state"],
        side=side,
        candidates=own_actions,
        opponent_candidates=opponent_actions,
        depth=depth,
        max_choices=simulator_max_choices,
        opponent_policy=simulator_opponent_policy,
        robust_worst_weight=simulator_robust_worst_weight,
    )
    matrix = np.asarray(result.get("values") or [], dtype=np.float32)
    if matrix.shape != (len(own_actions), len(opponent_actions)):
        matrix = np.zeros((len(own_actions), len(opponent_actions)), dtype=np.float32)
    action_utils = utilities_against_strategy(matrix, opponent_strategy, opponent_actions)
    current_value = float(
        sum(float(strategy.get(action, 0.0)) * float(value) for action, value in zip(own_actions, action_utils))
    )
    regrets = {
        action: float(value - current_value)
        for action, value in zip(own_actions, action_utils)
    }
    if neural_prior is not None and neural_examples is not None:
        neural_examples.append(
            neural_prior.example_from_utilities(
                snapshot,
                side,
                own_actions,
                action_utils,
                current_value,
                target_temperature=neural_target_temperature,
            )
        )
    model.update(key, own_actions, strategy, regrets)
    positive_regret = sum(max(0.0, value) for value in regrets.values())
    return (
        strategy,
        positive_regret,
        entropy(strategy),
        int(result.get("simulation_errors") or 0),
        int(result.get("skipped_branches") or 0),
    )


def limit_actions(actions: list[str], max_actions: int) -> list[str]:
    actions = [normalize_action(action) for action in actions]
    if max_actions > 0:
        return actions[:max_actions]
    return actions


def train_cfr_iteration(
    args: Any,
    model: CFRModel,
    *,
    iteration: int,
    opponent: str,
    neural_prior: CFRNeuralPrior | None = None,
    neural_examples: list[dict[str, Any]] | None = None,
) -> CFRIterationStats:
    battle_format = resolve_format(args.format)
    if not should_use_team(battle_format):
        raise ValueError("Offline CFR training requires a fixed team format")
    team_text = load_team(args.team)
    client = CFRShowdownClient(args.simulator_url, timeout=args.simulator_timeout)
    stats = CFRIterationStats(iteration=iteration, opponent=opponent)

    for game_index in range(args.games):
        rng = random.Random((iteration * 1000003) + game_index + int(args.seed))
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
        battle_tag = f"cfr-{iteration}-{game_index + 1}"

        for _ in range(max(1, int(args.offline_max_turns))):
            if snapshot.get("ended"):
                break
            legal = snapshot.get("legal") or {}
            p1_actions = limit_actions(list(legal.get("p1") or []), args.max_actions)
            p2_actions = limit_actions(list(legal.get("p2") or []), args.max_actions)
            p1_auto_default = is_auto_default(p1_actions)
            p2_auto_default = is_auto_default(p2_actions)
            p1_train_actions = [] if p1_auto_default else strategic_actions(p1_actions)
            p2_train_actions = [] if p2_auto_default else strategic_actions(p2_actions)
            if not p1_actions and not p2_actions:
                break

            p1_strategy: dict[str, float] = {}
            p2_strategy: dict[str, float] = {}
            if p2_train_actions:
                p2_strategy = (
                    model.strategy(offline_state_key(snapshot, "p2"), p2_train_actions)
                    if opponent == "self"
                    else baseline_strategy(opponent, p2_train_actions)
                )
            if p1_train_actions:
                p1_strategy, pos_regret, ent, errors, skipped = cfr_update(
                    model=model,
                    client=client,
                    snapshot=snapshot,
                    side="p1",
                    own_actions=p1_train_actions,
                    opponent_actions=p2_train_actions or [""],
                    opponent_strategy=p2_strategy or {"": 1.0},
                    depth=args.cfr_depth,
                    simulator_max_choices=args.simulator_max_choices,
                    simulator_opponent_policy=args.simulator_opponent_policy,
                    simulator_robust_worst_weight=args.simulator_robust_worst_weight,
                    neural_prior=neural_prior,
                    neural_examples=neural_examples,
                    neural_prior_weight=args.neural_prior_weight,
                    neural_target_temperature=args.neural_target_temperature,
                    min_average_visits=args.min_average_visits,
                )
                stats.positive_regret_sum += pos_regret
                stats.entropy_sum += ent
                stats.simulator_errors += errors
                stats.skipped_branches += skipped
                stats.regret_updates += 1

            if opponent == "self" and p2_train_actions:
                p1_for_p2 = p1_strategy or baseline_strategy("random", p1_train_actions or [""])
                _, pos_regret, ent, errors, skipped = cfr_update(
                    model=model,
                    client=client,
                    snapshot=snapshot,
                    side="p2",
                    own_actions=p2_train_actions,
                    opponent_actions=p1_train_actions or [""],
                    opponent_strategy=p1_for_p2,
                    depth=args.cfr_depth,
                    simulator_max_choices=args.simulator_max_choices,
                    simulator_opponent_policy=args.simulator_opponent_policy,
                    simulator_robust_worst_weight=args.simulator_robust_worst_weight,
                    neural_prior=neural_prior,
                    neural_examples=neural_examples,
                    neural_prior_weight=args.neural_prior_weight,
                    neural_target_temperature=args.neural_target_temperature,
                    min_average_visits=args.min_average_visits,
                )
                stats.positive_regret_sum += pos_regret
                stats.entropy_sum += ent
                stats.simulator_errors += errors
                stats.skipped_branches += skipped
                stats.regret_updates += 1

            choices: dict[str, str] = {}
            if p1_train_actions:
                choices["p1"] = sample_action(p1_strategy or baseline_strategy("random", p1_train_actions), rng)
            elif p1_auto_default:
                choices["p1"] = "default"
            if p2_train_actions:
                choices["p2"] = sample_action(p2_strategy or baseline_strategy("random", p2_train_actions), rng)
            elif p2_auto_default:
                choices["p2"] = "default"
            if not choices:
                break
            p1_logged_action = "" if choices.get("p1") == "default" else choices.get("p1", "")
            p2_logged_action = "" if choices.get("p2") == "default" else choices.get("p2", "")
            if p1_logged_action or p2_logged_action:
                append_jsonl(
                    args.rollout_path,
                    {
                        "schema_version": "cfr-rollout-v1",
                        "iteration": iteration,
                        "battle_tag": battle_tag,
                        "turn": int(snapshot.get("turn") or 0),
                        "opponent": opponent,
                        "p1_action": p1_logged_action,
                        "p2_action": p2_logged_action,
                        "p1_state_key": offline_state_key(snapshot, "p1"),
                        "p2_state_key": offline_state_key(snapshot, "p2"),
                        "p1_strategy": p1_strategy,
                        "p2_strategy": p2_strategy,
                    },
                )
                stats.decisions += 1
            snapshot = client.choose(
                state=snapshot["state"],
                choices=choices,
                max_choices=args.max_actions,
            )

        reward = outcome_for_side(snapshot, "p1")
        score = score_for_side(snapshot, "p1")
        truncated = not bool(snapshot.get("ended"))
        if reward == 0.0 and truncated:
            reward = 1.0 if score >= 0.0 else -1.0
        stats.games += 1
        stats.total_reward += reward
        stats.total_score += score
        stats.total_turns += int(snapshot.get("turn") or 0)
        if truncated:
            stats.truncated_games += 1
        if reward > 0:
            stats.wins += 1
            stats.score_adjusted_wins += 1
        elif reward < 0:
            stats.losses += 1
            stats.score_adjusted_losses += 1
        else:
            stats.draws += 1

        if (game_index + 1) % max(1, int(args.progress_games)) == 0:
            print(
                f"  CFR rollout progress: {game_index + 1}/{args.games} battle(s), "
                f"{stats.decisions} decisions",
                flush=True,
            )

    stats.finish()
    return stats


def stats_to_stage(stats: CFRIterationStats, started: int, model_stats: dict[str, Any]) -> dict[str, Any]:
    ended = started + stats.decisions
    return {
        "stage": stats.iteration,
        "started": started,
        "ended": ended,
        "duration": stats.decisions,
        "transition_reason": "iteration_finished",
        "n_episodes": stats.games,
        "win_rate": stats.win_rate,
        "score_adjusted_win_rate": stats.score_adjusted_win_rate,
        "avg_reward": stats.avg_reward,
        "avg_score": stats.avg_score,
        "avg_episode_length": stats.avg_turns,
        "final_loss": stats.avg_positive_regret,
        "final_policy_loss": stats.avg_entropy,
        "final_value_loss": None,
        "reward_breakdown": {
            "a.win": stats.wins * 15.0,
            "a.loss": stats.losses * -15.0,
            "a.draw": stats.draws * 0.0,
        },
        "n_updates": stats.regret_updates,
        "opponent": stats.opponent,
        "decisions": stats.decisions,
        "truncated_games": stats.truncated_games,
        "score_adjusted_wins": stats.score_adjusted_wins,
        "score_adjusted_losses": stats.score_adjusted_losses,
        "wall_time_sec": max(0.0, stats.ended_at - stats.started_at),
        "avg_positive_regret": stats.avg_positive_regret,
        "avg_strategy_entropy": stats.avg_entropy,
        "simulator": {
            "errors": stats.simulator_errors,
            "skipped_branches": stats.skipped_branches,
        },
        "model": model_stats,
    }


def aggregate_common(stages: list[dict[str, Any]], *, checkpoint_path: Path, log_path: Path, rollout_path: Path) -> dict[str, Any]:
    total_games = sum(int(stage.get("n_episodes") or 0) for stage in stages)
    wins = sum(round(float(stage.get("win_rate") or 0.0) * int(stage.get("n_episodes") or 0)) for stage in stages)
    losses = sum(int(abs(float((stage.get("reward_breakdown") or {}).get("a.loss", 0.0))) / 15.0) for stage in stages)
    draws = max(0, total_games - wins - losses)
    total_decisions = sum(int(stage.get("decisions") or stage.get("duration") or 0) for stage in stages)
    total_updates = sum(int(stage.get("n_updates") or 0) for stage in stages)
    truncated_games = sum(int(stage.get("truncated_games") or 0) for stage in stages)
    score_adjusted_wins = sum(int(stage.get("score_adjusted_wins") or 0) for stage in stages)
    score_adjusted_losses = sum(int(stage.get("score_adjusted_losses") or 0) for stage in stages)
    avg_reward = (
        sum(float(stage.get("avg_reward") or 0.0) * int(stage.get("n_episodes") or 0) for stage in stages)
        / total_games
        if total_games
        else 0.0
    )
    avg_len = (
        sum(float(stage.get("avg_episode_length") or 0.0) * int(stage.get("n_episodes") or 0) for stage in stages)
        / total_games
        if total_games
        else 0.0
    )
    avg_score = (
        sum(float(stage.get("avg_score") or 0.0) * int(stage.get("n_episodes") or 0) for stage in stages)
        / total_games
        if total_games
        else 0.0
    )
    final_stage = stages[-1] if stages else {}
    return {
        "schema_version": "common-v1",
        "model_family": "cfr_tabular",
        "algorithm": "cfr_tabular_approx",
        "total_games": total_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / total_games if total_games else 0.0,
        "score_adjusted_win_rate": score_adjusted_wins / total_games if total_games else 0.0,
        "score_adjusted_wins": score_adjusted_wins,
        "score_adjusted_losses": score_adjusted_losses,
        "truncated_games": truncated_games,
        "avg_episode_length": avg_len,
        "avg_reward": avg_reward,
        "avg_score": avg_score,
        "final_loss": final_stage.get("final_loss"),
        "final_policy_loss": final_stage.get("final_policy_loss"),
        "final_value_loss": final_stage.get("final_value_loss"),
        "final_entropy": final_stage.get("avg_strategy_entropy"),
        "total_updates": total_updates,
        "total_decisions": total_decisions,
        "total_iterations": len(stages),
        "completed_iterations": len(stages),
        "last_iteration": final_stage.get("stage", 0),
        "training_log_path": str(log_path),
        "rollout_path": str(rollout_path),
        "checkpoint_path": str(checkpoint_path),
    }


def summarize_opponents(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_opponent: dict[str, Counter] = {}
    for stage in stages:
        opponent = str(stage.get("opponent") or "unknown")
        item = by_opponent.setdefault(opponent, Counter())
        item["games"] += int(stage.get("n_episodes") or 0)
        item["wins"] += round(float(stage.get("win_rate") or 0.0) * int(stage.get("n_episodes") or 0))
        item["decisions"] += int(stage.get("decisions") or 0)
    return [
        {
            "opponent": opponent,
            "games": int(counts["games"]),
            "wins": int(counts["wins"]),
            "win_rate": float(counts["wins"] / counts["games"]) if counts["games"] else 0.0,
            "decisions": int(counts["decisions"]),
        }
        for opponent, counts in sorted(by_opponent.items())
    ]
