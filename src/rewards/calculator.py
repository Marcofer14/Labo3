"""
src.rewards.calculator · Orquestador de las 4 capas.

Uso:
    calc = RewardCalculator(RewardConfig.stage_1(), data, team_stats)
    reward, breakdown = calc.compute(battle, last_actions=(act_a, act_b))

`last_actions` es opcional; si está en None, los componentes que
dependen de la acción (Layer B-extra, C, D) caen al valor 0.
"""

from __future__ import annotations
from typing import Optional

from poke_env.battle.double_battle import DoubleBattle

from src.rewards.config         import RewardConfig
from src.rewards.state          import (
    BattleStateTracker,
    TurnSnapshot,
    snapshot_battle,
)
from src.rewards.action_decoder import DecodedAction
from src.rewards import layer_a, layer_b, layer_c, layer_d


class RewardCalculator:
    """Orquestador estatal del cálculo de reward."""

    def __init__(
        self,
        config:       RewardConfig,
        data:         dict,             # load_all_data()
        team_stats:   dict,             # {species: stats_dict} de nuestro equipo
    ):
        self.config     = config
        self.data       = data
        self.team_stats = team_stats
        self.tracker    = BattleStateTracker()

        # Acumulado por episodio para métricas
        self._episode_breakdown: dict[str, dict[str, float]] = {}

    def set_config(self, cfg: RewardConfig) -> None:
        """Permite al curriculum scheduler cambiar pesos/flags en caliente."""
        self.config = cfg

    def reset_episode(self, tag: str) -> None:
        self.tracker.clear(tag)
        self._episode_breakdown.pop(tag, None)

    def get_episode_breakdown(self, tag: str) -> dict[str, float]:
        """Suma acumulada por componente para esta batalla."""
        return dict(self._episode_breakdown.get(tag, {}))

    def compute(
        self,
        battle:       DoubleBattle,
        last_actions: Optional[tuple[DecodedAction, DecodedAction]] = None,
    ) -> tuple[float, dict[str, float]]:
        tag  = battle.battle_tag
        prev = self.tracker.previous(tag)
        curr = snapshot_battle(battle, self.team_stats, self.data["pokemon"])

        total = 0.0
        breakdown: dict[str, float] = {}

        v, b = layer_a.compute(curr, prev, self.config)
        total += v; breakdown.update(b)

        v, b = layer_b.compute(curr, prev, last_actions, self.data, self.config)
        total += v; breakdown.update(b)

        v, b = layer_c.compute(curr, prev, last_actions, self.data, self.config)
        # Aplicar el ramp-up gradual de Layer C
        scale = float(self.config.layer_c_scale)
        if scale != 1.0:
            v = v * scale
            b = {k: val * scale for k, val in b.items()}
        total += v; breakdown.update(b)

        v, b = layer_d.compute(curr, prev, last_actions, self.data, self.config)
        total += v; breakdown.update(b)

        # Acumular para el episodio
        ep = self._episode_breakdown.setdefault(tag, {})
        for k, val in breakdown.items():
            ep[k] = ep.get(k, 0.0) + val
        ep["_total"] = ep.get("_total", 0.0) + total

        if curr.finished:
            # NO limpiamos el breakdown todavía: el callback puede leerlo
            self.tracker.clear(tag)
        else:
            self.tracker.update(tag, curr)

        return total, breakdown
