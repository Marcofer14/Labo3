"""
src.training — pipeline de entrenamiento del VGC bot.

Componentes:
  config.py     · TrainingConfig (hiperparámetros + arquitectura)
  policy.py     · MaskableRecurrentLstmPolicy (LSTM + ReLU + masking)
  curriculum.py · CurriculumScheduler con detección de plateau de loss
  league.py     · SelfPlayLeague con eviction por win-rate
  callbacks.py  · Métricas: loss, activaciones, reward breakdown, win rate
  report.py     · Generador de informe HTML + PNG por fase
"""

from src.training.config     import TrainingConfig
from src.training.curriculum import CurriculumScheduler, StageMetrics
from src.training.league     import SelfPlayLeague, LeagueEntry
from src.training.callbacks  import (
    LossPlateauCallback,
    RewardBreakdownCallback,
    ActivationStatsCallback,
    WinRateCallback,
    PhaseLogCallback,
    SnapshotLeagueCallback,
    LeagueResultCallback,
)
from src.training.opponents  import LeagueOpponent, action_int_to_order, actions_to_double_order
from src.training.tournament import run_tournament
from src.training.report     import generate_final_report

__all__ = [
    "TrainingConfig",
    "CurriculumScheduler",
    "StageMetrics",
    "SelfPlayLeague",
    "LeagueEntry",
    "LossPlateauCallback",
    "RewardBreakdownCallback",
    "ActivationStatsCallback",
    "WinRateCallback",
    "PhaseLogCallback",
    "SnapshotLeagueCallback",
    "LeagueResultCallback",
    "LeagueOpponent",
    "action_int_to_order",
    "actions_to_double_order",
    "run_tournament",
    "generate_final_report",
]
