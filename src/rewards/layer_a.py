"""
src.rewards.layer_a · Recompensas terminales (win / loss).

Activado en todos los stages.
"""

from __future__ import annotations
from typing import Optional

from src.rewards.state  import TurnSnapshot
from src.rewards.config import RewardConfig


def compute(
    curr: TurnSnapshot,
    prev: Optional[TurnSnapshot],
    cfg:  RewardConfig,
) -> tuple[float, dict[str, float]]:
    if not cfg.enable_layer_a or not curr.finished:
        return 0.0, {}
    if curr.won:
        return  cfg.w_win, {"a.win":  cfg.w_win}
    return -cfg.w_win, {"a.loss": -cfg.w_win}
