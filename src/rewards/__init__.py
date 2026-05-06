"""
src.rewards — sistema modular de recompensas en 4 capas.

Capa A — terminal      (win/loss)            Stage 1+
Capa B — táctico       (cada turno)          Stage 1+
Capa C — estratégico   (decisiones de campo) Stage 2+
Capa D — meta/endgame  (visión de partida)   Stage 3+

Uso:
    from src.rewards import RewardCalculator, RewardConfig

    calc = RewardCalculator(RewardConfig.stage_1(), data)
    reward, breakdown = calc.compute(battle, last_actions)
    # breakdown es {"b.dmg_dealt": +12.4, "b.ko": +3.0, ...}
"""

from src.rewards.config     import RewardConfig
from src.rewards.calculator import RewardCalculator
from src.rewards.state      import BattleStateTracker, TurnSnapshot, PokemonSnapshot, FieldSnapshot
from src.rewards.action_decoder import DecodedAction, decode_action

__all__ = [
    "RewardConfig",
    "RewardCalculator",
    "BattleStateTracker",
    "TurnSnapshot",
    "PokemonSnapshot",
    "FieldSnapshot",
    "DecodedAction",
    "decode_action",
]
