"""poke-env player that chooses moves with AlphaZero-style MCTS."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from poke_env.player import Player
from poke_env.player.battle_order import DoubleBattleOrder

from src.alphazero.mcts import AlphaZeroMCTS, MCTSConfig, heuristic_candidate_value
from src.alphazero.network import AlphaZeroPolicyValueNet, load_checkpoint
from src.alphazero.showdown_simulator import ShowdownSimulationTracker, ShowdownSimulatorClient


class AlphaZeroMCTSPlayer(Player):
    """Candidate-ranker player for VGC doubles.

    The player loads a policy/value checkpoint when one is available. Without a
    checkpoint it still runs, using a randomly initialized network plus a small
    tactical heuristic, which is useful for smoke tests before training.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        simulations: int = 64,
        search_depth: int = 1,
        max_candidates: int = 96,
        cpuct: float = 1.5,
        temperature: float = 0.0,
        heuristic_weight: float = 0.75,
        depth2_weight: float = 0.65,
        showdown_simulator_url: str | None = None,
        live_state_url: str | None = None,
        simulation_tracker: ShowdownSimulationTracker | None = None,
        simulator_timeout: float = 10.0,
        simulator_max_choices: int = 12,
        simulator_opponent_policy: str = "robust",
        simulator_robust_worst_weight: float = 0.35,
        require_showdown_simulator: bool = False,
        device: str = "cpu",
        record_decisions: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        if self.checkpoint_path and self.checkpoint_path.exists():
            self.model = load_checkpoint(self.checkpoint_path, device=device)
        else:
            self.model = AlphaZeroPolicyValueNet().to(device)
            self.model.eval()

        simulator = None
        if showdown_simulator_url:
            simulator = ShowdownSimulatorClient(
                showdown_simulator_url,
                live_state_url=live_state_url,
                timeout=simulator_timeout,
                max_choices=simulator_max_choices,
                opponent_policy=simulator_opponent_policy,
                robust_worst_weight=simulator_robust_worst_weight,
            )

        self.mcts = AlphaZeroMCTS(
            self.model,
            MCTSConfig(
                simulations=simulations,
                depth=search_depth,
                cpuct=cpuct,
                temperature=temperature,
                heuristic_weight=heuristic_weight,
                depth2_weight=depth2_weight,
                device=device,
                showdown_simulator=simulator,
                simulation_tracker=simulation_tracker,
                require_showdown_simulator=require_showdown_simulator,
            ),
        )
        self.max_candidates = max_candidates
        self.require_showdown_simulator = require_showdown_simulator
        self.record_decisions = record_decisions
        self.decision_log: list[dict[str, Any]] = []

    def _legal_candidates(self, battle) -> list[Any]:
        try:
            candidates = DoubleBattleOrder.join_orders(*battle.valid_orders)
        except Exception:
            candidates = []
        if not candidates:
            return [Player.choose_random_doubles_move(battle)]

        max_candidates = int(self.max_candidates or 0)
        if max_candidates > 0 and len(candidates) > max_candidates:
            candidates = sorted(
                candidates,
                key=lambda order: heuristic_candidate_value(battle, order),
                reverse=True,
            )[:max_candidates]
        return candidates

    def choose_move(self, battle):
        candidates = self._legal_candidates(battle)
        try:
            result = self.mcts.search(battle, candidates)
        except Exception as exc:
            if self.require_showdown_simulator:
                raise
            print(f"  Aviso AlphaZero: fallback random por error en MCTS: {exc}", flush=True)
            return Player.choose_random_doubles_move(battle)

        if self.record_decisions:
            self.decision_log.append(
                {
                    "battle_tag": getattr(battle, "battle_tag", ""),
                    "turn": int(getattr(battle, "turn", 0) or 0),
                    "candidate_count": len(candidates),
                    "forced_decision": len(candidates) <= 1,
                    "candidate_messages": [
                        getattr(order, "message", None) or str(order) for order in candidates
                    ],
                    "state_features": result.state_features.astype(np.float32).tolist(),
                    "action_features": result.action_features.astype(np.float32).tolist(),
                    "selected_index": int(result.selected_index),
                    "visit_probs": result.visit_probs.astype(np.float32).tolist(),
                    "priors": result.priors.astype(np.float32).tolist(),
                    "candidate_values": result.candidate_values.astype(np.float32).tolist(),
                    "old_logprob": float(result.logprob),
                    "old_value": float(result.value),
                    "simulator_used": bool(result.simulator_used),
                    "simulator_repairs": int(result.simulator_repairs),
                    "simulator_errors": int(result.simulator_errors),
                    "simulator_skipped_branches": int(result.simulator_skipped_branches),
                    "simulator_error_details": result.simulator_error_details or [],
                    "simulator_error_stage_counts": result.simulator_error_stage_counts or {},
                    "selected_message": getattr(result.order, "message", None)
                    or str(result.order),
                }
            )
            if len(self.decision_log) % 25 == 0:
                print(
                    f"  AlphaZero rollout progress: {len(self.decision_log)} decisions collected",
                    flush=True,
                )
        return result.order
