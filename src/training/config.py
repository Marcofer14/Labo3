"""
src.training.config · TrainingConfig central.

Todos los hiperparámetros de PPO + arquitectura + curriculum.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    # ── PPO hiperparámetros ─────────────────────────────────────
    # Learning rate con schedule lineal 3e-4 → 1e-5
    lr_initial:       float = 3e-4
    lr_final:         float = 1e-5

    # Paralelización: 4 envs × 1024 steps = rollout buffer 4096
    n_envs:           int   = 4
    n_steps:          int   = 1024
    batch_size:       int   = 256
    n_epochs:         int   = 4

    gamma:            float = 0.99
    gae_lambda:       float = 0.95
    clip_range:       float = 0.2
    clip_range_vf:    float | None = None
    ent_coef:         float = 0.01     # arranque con algo de exploración
    vf_coef:          float = 0.5
    max_grad_norm:    float = 0.5

    # ── Arquitectura ────────────────────────────────────────────
    # MLP feature extractor antes del LSTM
    net_arch:         list[int] = field(default_factory=lambda: [256])
    activation:       str       = "relu"      # "relu" | "tanh"
    lstm_hidden_size: int       = 128
    lstm_layers:      int       = 1
    shared_lstm:      bool      = False
    enable_critic_lstm: bool    = True

    # ── Algoritmo ───────────────────────────────────────────────
    # "recurrent_ppo" (LSTM) | "maskable_ppo" (sin LSTM, con mask)
    algorithm:        str = "recurrent_ppo"

    # ── Curriculum ──────────────────────────────────────────────
    # Detección de plateau: la loss "no enseña nada" si la mejora
    # relativa promedio sobre `plateau_window` updates es < `plateau_eps`
    plateau_window:        int   = 30      # número de updates a mirar
    plateau_eps:           float = 0.005   # 0.5% de mejora relativa mínima
    stage_min_timesteps:   int   = 80_000  # mínimo por stage antes de poder transicionar
    stage_max_timesteps:   int   = 600_000 # corte duro si nunca platea
    layer_c_ramp_steps:    int   = 100_000 # cuántos pasos para subir layer_c_scale 0→1

    # ── Self-play league ────────────────────────────────────────
    league_max_size:       int   = 6       # cuántos snapshots conservar
    league_snapshot_every: int   = 100_000
    league_eviction:       str   = "lowest_winrate"  # mantenemos los más duros
    league_min_battles_for_eviction: int = 20

    # ── Métricas ────────────────────────────────────────────────
    activation_log_every:  int   = 5       # cada N updates registrar activations
    win_rate_window:       int   = 100     # rolling window de batallas

    # ── Logging / checkpoints ───────────────────────────────────
    total_timesteps:       int   = 2_000_000
    checkpoint_every:      int   = 50_000
    eval_every:            int   = 100_000
    eval_n_battles:        int   = 30
    seed:                  int   = 42

    # ── Paths ───────────────────────────────────────────────────
    log_dir:        str = "logs"
    checkpoint_dir: str = "checkpoints"
    report_dir:     str = "reports"
    rivalteams_dir: str = "rivalteams"

    # ── Helpers ─────────────────────────────────────────────────

    def linear_lr_schedule(self):
        """Devuelve un callable schedule(progress_remaining) → lr."""
        a, b = self.lr_initial, self.lr_final
        def _sched(progress_remaining: float) -> float:
            return b + (a - b) * progress_remaining
        return _sched

    def rollout_buffer_size(self) -> int:
        return self.n_envs * self.n_steps

    def minibatches_per_epoch(self) -> int:
        return self.rollout_buffer_size() // self.batch_size
