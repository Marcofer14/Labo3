"""
src.training.curriculum · Scheduler de stages con detección de plateau.

Filosofía:
  · La transición entre stages NO se decide por timesteps fijos sino por
    plateau de la loss. Cuando la pérdida deja de bajar significativamente
    (improvement relativo < eps en una ventana de N updates), pasamos al
    siguiente stage.
  · Hay un mínimo de timesteps por stage (`stage_min_timesteps`) para evitar
    transiciones tempranas por ruido del rollout.
  · Hay un máximo (`stage_max_timesteps`) como cinturón de seguridad.

Stages:
  1 — Layer A + B básico                  vs RandomPlayer
  2 — Layer A + B completo (B-extra)      vs MaxBasePower / heurístico simple
  3 — Layer A + B + C (con ramp gradual)  vs heurístico fuerte / pool de teams
  4 — Layer A + B + C + D                 vs self-play league
  5 — full + league completo              ladder / eval
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable
import math

from src.rewards.config import RewardConfig
from src.training.config import TrainingConfig


@dataclass
class StageMetrics:
    """Estadísticas acumuladas por stage para el reporte final."""
    stage:               int
    started_timestep:    int
    ended_timestep:      int  = -1
    transition_reason:   str  = ""           # "plateau" | "max_timesteps" | "manual"

    # series temporales (una entrada por update)
    losses_value:        list[float] = field(default_factory=list)
    losses_policy:       list[float] = field(default_factory=list)
    losses_entropy:      list[float] = field(default_factory=list)
    approx_kl:           list[float] = field(default_factory=list)
    clip_fractions:      list[float] = field(default_factory=list)
    explained_variance:  list[float] = field(default_factory=list)

    # episode-level
    episode_rewards:     list[float] = field(default_factory=list)
    episode_lengths:     list[int]   = field(default_factory=list)
    episode_won:         list[bool]  = field(default_factory=list)

    # reward breakdown acumulado por componente
    reward_breakdown:    dict[str, float] = field(default_factory=dict)

    @property
    def duration_timesteps(self) -> int:
        if self.ended_timestep < 0:
            return 0
        return self.ended_timestep - self.started_timestep

    @property
    def n_episodes(self) -> int:
        return len(self.episode_won)

    @property
    def win_rate(self) -> float:
        return (sum(self.episode_won) / self.n_episodes) if self.episode_won else 0.0

    @property
    def avg_reward(self) -> float:
        return sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0.0

    @property
    def avg_episode_length(self) -> float:
        return sum(self.episode_lengths) / len(self.episode_lengths) if self.episode_lengths else 0.0


class CurriculumScheduler:
    """
    Mantiene el stage actual, detecta plateau, gestiona ramp de Layer C.

    Uso (desde un callback de SB3):
        sched = CurriculumScheduler(train_cfg)
        sched.start(current_timestep=0)
        # ... después de cada update PPO
        sched.record_loss(value_loss=..., policy_loss=..., timestep=...)
        if sched.should_advance():
            sched.advance(timestep=...)
        cfg = sched.current_reward_config(timestep)   # con layer_c_scale
    """

    def __init__(self, train_cfg: TrainingConfig):
        self.cfg = train_cfg

        # ── Estado del scheduler ─────────────────────────────────
        self.current_stage: int = 1
        self.history: list[StageMetrics] = []
        self.current_metrics: Optional[StageMetrics] = None

        # Para plateau detection
        self._loss_window: list[float] = []

        # Marca dónde entramos a stage 3 (para el ramp de Layer C)
        self._stage3_start_timestep: Optional[int] = None

    # ── API principal ────────────────────────────────────────────

    def start(self, current_timestep: int = 0, stage: int = 1) -> StageMetrics:
        self.current_stage = int(stage)
        if self.current_stage >= 3:
            self._stage3_start_timestep = max(
                0,
                current_timestep - self.cfg.layer_c_ramp_steps,
            )
        m = StageMetrics(stage=self.current_stage, started_timestep=current_timestep)
        self.current_metrics = m
        self.history.append(m)
        return m

    def record_loss(
        self,
        value_loss:         float,
        policy_loss:        float,
        entropy_loss:       float,
        approx_kl:          float,
        clip_fraction:      float,
        explained_variance: float,
        timestep:           int,
    ) -> None:
        m = self.current_metrics
        if m is None:
            return
        m.losses_value.append(float(value_loss))
        m.losses_policy.append(float(policy_loss))
        m.losses_entropy.append(float(entropy_loss))
        m.approx_kl.append(float(approx_kl))
        m.clip_fractions.append(float(clip_fraction))
        m.explained_variance.append(float(explained_variance))
        # Plateau usa value_loss como señal principal (suele ser la más estable)
        self._loss_window.append(float(value_loss))
        if len(self._loss_window) > self.cfg.plateau_window:
            self._loss_window.pop(0)

    def record_episode(
        self,
        reward:    float,
        length:    int,
        won:       bool,
        breakdown: dict[str, float] | None = None,
    ) -> None:
        m = self.current_metrics
        if m is None:
            return
        m.episode_rewards.append(float(reward))
        m.episode_lengths.append(int(length))
        m.episode_won.append(bool(won))
        if breakdown:
            for k, v in breakdown.items():
                m.reward_breakdown[k] = m.reward_breakdown.get(k, 0.0) + float(v)

    def should_advance(self, current_timestep: int) -> tuple[bool, str]:
        """
        Devuelve (debe_transicionar, razón).
        """
        if self.current_stage >= 5:
            return False, ""
        m = self.current_metrics
        if m is None:
            return False, ""
        elapsed = current_timestep - m.started_timestep
        if elapsed < self.cfg.stage_min_timesteps:
            return False, ""
        if elapsed >= self.cfg.stage_max_timesteps:
            return True, "max_timesteps"
        if self._is_loss_plateau():
            return True, "plateau"
        return False, ""

    def advance(self, timestep: int, reason: str = "plateau") -> int:
        """Cierra el stage actual y abre el siguiente. Devuelve el nuevo stage."""
        if self.current_metrics is not None:
            self.current_metrics.ended_timestep = timestep
            self.current_metrics.transition_reason = reason
        self.current_stage += 1
        self._loss_window.clear()
        if self.current_stage == 3:
            self._stage3_start_timestep = timestep
        new_m = StageMetrics(stage=self.current_stage, started_timestep=timestep)
        self.current_metrics = new_m
        self.history.append(new_m)
        return self.current_stage

    def finalize(self, timestep: int) -> None:
        """Cierra el stage actual al final del run."""
        if self.current_metrics is not None and self.current_metrics.ended_timestep < 0:
            self.current_metrics.ended_timestep = timestep
            if not self.current_metrics.transition_reason:
                self.current_metrics.transition_reason = "end_of_training"

    # ── Reward config para el stage actual ──────────────────────

    def current_reward_config(self, current_timestep: int) -> RewardConfig:
        s = self.current_stage
        if s == 1:
            return RewardConfig.stage_1()
        if s == 2:
            return RewardConfig.stage_2()
        if s == 3:
            return RewardConfig.stage_3(layer_c_scale=self._layer_c_scale(current_timestep))
        if s == 4:
            return RewardConfig.stage_4(layer_c_scale=1.0)
        return RewardConfig.stage_5(layer_c_scale=1.0)

    def _layer_c_scale(self, current_timestep: int) -> float:
        """0 → 1 lineal en `layer_c_ramp_steps` desde la entrada a stage 3."""
        if self._stage3_start_timestep is None:
            return 0.0
        elapsed = current_timestep - self._stage3_start_timestep
        if elapsed <= 0:
            return 0.0
        ramp = self.cfg.layer_c_ramp_steps
        if ramp <= 0:
            return 1.0
        return float(min(1.0, elapsed / ramp))

    # ── Plateau detection ───────────────────────────────────────

    def _is_loss_plateau(self) -> bool:
        w = self._loss_window
        if len(w) < self.cfg.plateau_window:
            return False
        # Mejora relativa: comparamos primera mitad vs segunda mitad de la ventana
        half = len(w) // 2
        avg_old = sum(w[:half]) / half
        avg_new = sum(w[half:]) / (len(w) - half)
        if avg_old == 0 or math.isnan(avg_old):
            return False
        rel_improve = (avg_old - avg_new) / abs(avg_old)
        return rel_improve < self.cfg.plateau_eps
