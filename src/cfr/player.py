"""poke-env player backed by a tabular CFR average strategy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from poke_env.player import Player
from poke_env.player.battle_order import DoubleBattleOrder

from src.alphazero.mcts import heuristic_candidate_value
from src.cfr.model import CFRModel, normalize_action
from src.cfr.neural import CFRNeuralPrior, mix_strategies
from src.cfr.state_abstraction import battle_state_key


class CFRPlayer(Player):
    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        max_candidates: int = 32,
        temperature: float = 0.0,
        fallback: str = "heuristic",
        neural_checkpoint_path: str | Path | None = None,
        neural_weight: float = 0.70,
        min_average_visits: int = 3,
        neural_device: str = "cpu",
        record_decisions: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.model = CFRModel.load(self.checkpoint_path) if self.checkpoint_path and self.checkpoint_path.exists() else CFRModel()
        self.max_candidates = int(max_candidates or 0)
        self.temperature = float(temperature)
        self.fallback = fallback
        self.neural_checkpoint_path = Path(neural_checkpoint_path) if neural_checkpoint_path else None
        self.neural_prior = (
            CFRNeuralPrior(checkpoint_path=self.neural_checkpoint_path, device=neural_device)
            if self.neural_checkpoint_path and self.neural_checkpoint_path.exists()
            else None
        )
        self.neural_weight = float(neural_weight)
        self.min_average_visits = int(min_average_visits)
        self.record_decisions = record_decisions
        self.decision_log: list[dict[str, Any]] = []

    def _legal_candidates(self, battle) -> list[Any]:
        try:
            candidates = DoubleBattleOrder.join_orders(*battle.valid_orders)
        except Exception:
            candidates = []
        if not candidates:
            return [Player.choose_random_doubles_move(battle)]
        if self.max_candidates > 0 and len(candidates) > self.max_candidates:
            candidates = sorted(
                candidates,
                key=lambda order: heuristic_candidate_value(battle, order),
                reverse=True,
            )[: self.max_candidates]
        return candidates

    def _fallback_order(self, battle, candidates: list[Any]) -> Any:
        if not candidates:
            return Player.choose_random_doubles_move(battle)
        if self.fallback == "random":
            return Player.choose_random_doubles_move(battle)
        return max(candidates, key=lambda order: heuristic_candidate_value(battle, order))

    def choose_move(self, battle):
        candidates = self._legal_candidates(battle)
        action_to_order = {
            normalize_action(getattr(order, "message", None) or str(order)): order
            for order in candidates
        }
        actions = list(action_to_order)
        if not actions:
            return Player.choose_random_doubles_move(battle)

        key = battle_state_key(battle, "p1")
        node = self.model.nodes.get(key)
        if node is None or node.visits <= 0:
            strategy = {}
        elif node.visits < max(1, self.min_average_visits):
            strategy = self.model.strategy(key, actions, average=False)
        else:
            strategy = self.model.strategy(key, actions, average=True)

        if self.neural_prior is not None and self.neural_prior.ready:
            prior_strategy, _ = self.neural_prior.predict_battle(battle, actions, candidates)
            if not strategy:
                strategy = prior_strategy
            else:
                weight = self.neural_weight
                if node is not None and node.visits < max(1, self.min_average_visits):
                    weight = max(weight, 1.0 - node.visits / max(1, self.min_average_visits))
                strategy = mix_strategies(actions, strategy, prior_strategy, weight)

        if not strategy:
            order = self._fallback_order(battle, candidates)
            selected = normalize_action(getattr(order, "message", None) or str(order))
        else:
            if self.temperature <= 0:
                selected = max(actions, key=lambda action: strategy.get(action, 0.0))
            else:
                import random

                threshold = random.random()
                cumulative = 0.0
                selected = actions[-1]
                for action in actions:
                    cumulative += float(strategy.get(action, 0.0))
                    if threshold <= cumulative:
                        selected = action
                        break
            order = action_to_order.get(selected) or self._fallback_order(battle, candidates)

        if self.record_decisions:
            self.decision_log.append(
                {
                    "battle_tag": getattr(battle, "battle_tag", ""),
                    "turn": int(getattr(battle, "turn", 0) or 0),
                    "state_key": key,
                    "candidate_count": len(candidates),
                    "selected_action": selected,
                    "strategy": strategy,
                    "checkpoint": str(self.checkpoint_path or ""),
                    "neural_checkpoint": str(self.neural_checkpoint_path or ""),
                }
            )
        return order
