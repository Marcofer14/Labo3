"""
src.training.callbacks · Callbacks de SB3 para tracking detallado.

  · LossPlateauCallback     · feed loss al curriculum scheduler + advance stage
  · RewardBreakdownCallback · suma reward por componente y por episodio
  · ActivationStatsCallback · mean/std/saturated por capa cada N updates
  · WinRateCallback         · rolling win rate por oponente
  · PhaseLogCallback        · logging por fase a TensorBoard
"""

from __future__ import annotations

from collections import deque
from typing import Any, Optional
import numpy as np

from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback


def _as_sequence(value) -> list:
    """Normaliza valores de SB3 que pueden venir como list, tuple, ndarray o None."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


# ── 1. LossPlateauCallback ───────────────────────────────────────

class LossPlateauCallback(BaseCallback):
    """
    Después de cada update de PPO:
      1. Lee las losses del logger de SB3
      2. Las pasa al CurriculumScheduler
      3. Si scheduler dice "advance", llama on_stage_advance(scheduler, new_stage)

    on_stage_advance es responsable de cambiar el RewardConfig en TODOS los envs
    del VecEnv y opcionalmente cambiar el oponente.
    """

    def __init__(
        self,
        scheduler,
        on_stage_advance,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.scheduler = scheduler
        self.on_stage_advance = on_stage_advance

    def _read_loss(self, key: str) -> Optional[float]:
        """Lee el valor más reciente de un key del logger de SB3."""
        try:
            val = self.logger.name_to_value.get(key)
            return float(val) if val is not None else None
        except Exception:
            return None

    def _on_rollout_end(self) -> None:
        # Después de update PPO, las losses están en el logger
        vl  = self._read_loss("train/value_loss")
        pl  = self._read_loss("train/policy_gradient_loss") or self._read_loss("train/policy_loss")
        el  = self._read_loss("train/entropy_loss")
        kl  = self._read_loss("train/approx_kl")
        cf  = self._read_loss("train/clip_fraction")
        ev  = self._read_loss("train/explained_variance")

        if vl is None or pl is None:
            return

        self.scheduler.record_loss(
            value_loss         = vl,
            policy_loss        = pl,
            entropy_loss       = el or 0.0,
            approx_kl          = kl or 0.0,
            clip_fraction      = cf or 0.0,
            explained_variance = ev or 0.0,
            timestep           = self.num_timesteps,
        )

        advance, reason = self.scheduler.should_advance(self.num_timesteps)
        if advance:
            new_stage = self.scheduler.advance(self.num_timesteps, reason=reason)
            if self.verbose:
                print(f"\n[curriculum] STAGE → {new_stage}  (reason: {reason}, t={self.num_timesteps:,})")
            try:
                self.on_stage_advance(self.scheduler, new_stage)
            except Exception as ex:
                if self.verbose:
                    print(f"  [curriculum] on_stage_advance fallo: {ex}")

    def _on_step(self) -> bool:
        return True


# ── 2. RewardBreakdownCallback ───────────────────────────────────

class RewardBreakdownCallback(BaseCallback):
    """
    Acumula el reward por componente sumando los breakdowns que vienen
    en `info["reward_breakdown"]` de cada env. Al cierre del episodio
    los flushea al scheduler y al logger.
    """

    def __init__(self, scheduler, log_every: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.scheduler = scheduler
        self.log_every = log_every
        # Acumulador por env
        self._accum: dict[int, dict[str, float]] = {}
        self._ep_steps: dict[int, int] = {}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", []) or []
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])

        for i, info in enumerate(infos):
            self._accum.setdefault(i, {})
            self._ep_steps[i] = self._ep_steps.get(i, 0) + 1

            bd = info.get("reward_breakdown") or {}
            for k, v in bd.items():
                self._accum[i][k] = self._accum[i].get(k, 0.0) + float(v)

            if i < len(dones) and dones[i]:
                # Episodio cerró → registrar
                ep_reward = float(info.get("episode", {}).get("r", sum(self._accum[i].values())))
                ep_length = int(info.get("episode", {}).get("l", self._ep_steps[i]))
                won = bool(info.get("won", info.get("is_success", False)))

                self.scheduler.record_episode(
                    reward    = ep_reward,
                    length    = ep_length,
                    won       = won,
                    breakdown = self._accum[i],
                )

                # Logger
                for k, v in self._accum[i].items():
                    self.logger.record(f"reward/{k}", v)

                self._accum[i] = {}
                self._ep_steps[i] = 0
        return True


# ── 3. ActivationStatsCallback ───────────────────────────────────

class ActivationStatsCallback(BaseCallback):
    """Loggea estadísticas de activaciones cada N updates."""

    def __init__(self, recorder, log_every: int = 5, verbose: int = 0):
        super().__init__(verbose)
        self.recorder = recorder
        self.log_every = log_every
        self._update_count = 0

    def _on_training_start(self) -> None:
        self.recorder.attach(self.model)

    def _on_training_end(self) -> None:
        self.recorder.detach()

    def _on_rollout_end(self) -> None:
        self._update_count += 1
        if self._update_count % self.log_every != 0:
            return
        stats = self.recorder.drain()
        if not stats:
            return
        # Agrupar por layer
        by_layer: dict[int, list[dict]] = {}
        for s in stats:
            by_layer.setdefault(s["layer"], []).append(s)
        for layer, lst in by_layer.items():
            self.logger.record(f"activations/L{layer}/mean",   np.mean([s["mean"] for s in lst]))
            self.logger.record(f"activations/L{layer}/std",    np.mean([s["std"] for s in lst]))
            self.logger.record(f"activations/L{layer}/zero",   np.mean([s["frac_zero"] for s in lst]))
            self.logger.record(f"activations/L{layer}/sat",    np.mean([s["frac_saturated"] for s in lst]))

    def _on_step(self) -> bool:
        return True


# ── 4. WinRateCallback ───────────────────────────────────────────

class WinRateCallback(BaseCallback):
    """Rolling win-rate sobre las últimas N batallas."""

    def __init__(self, window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window = window
        self._results = deque(maxlen=window)
        # Por oponente (key viene del info)
        self._per_opp: dict[str, deque] = {}

    def _on_step(self) -> bool:
        infos = _as_sequence(self.locals.get("infos"))
        dones = _as_sequence(self.locals.get("dones"))
        for i, info in enumerate(infos):
            if i >= len(dones) or not dones[i]:
                continue
            won = bool(info.get("won", info.get("is_success", False)))
            self._results.append(int(won))
            opp = info.get("opponent_label", "default")
            dq  = self._per_opp.setdefault(opp, deque(maxlen=self.window))
            dq.append(int(won))
        if self._results:
            self.logger.record("eval/winrate_rolling", sum(self._results) / len(self._results))
        for opp, dq in self._per_opp.items():
            if dq:
                self.logger.record(f"eval/winrate_{opp}", sum(dq) / len(dq))
        return True


# ── 5. PhaseLogCallback ──────────────────────────────────────────

class PhaseLogCallback(BaseCallback):
    """Loggea el stage actual + layer_c_scale en cada step."""

    def __init__(self, scheduler, verbose: int = 0):
        super().__init__(verbose)
        self.scheduler = scheduler

    def _on_step(self) -> bool:
        self.logger.record("curriculum/stage", self.scheduler.current_stage)
        self.logger.record("curriculum/layer_c_scale",
                           self.scheduler._layer_c_scale(self.num_timesteps))
        return True


# ── 6. SnapshotLeagueCallback ────────────────────────────────────

class SnapshotLeagueCallback(BaseCallback):
    """
    Cada `every` timesteps:
      1. guarda el modelo en `snapshot_dir / vgc_t<N>.zip`
      2. lo agrega al league (con eviction automático)
    """

    def __init__(
        self,
        league,
        scheduler,
        every:        int,
        snapshot_dir: str | Path,
        min_stage:    int  = 3,         # arrancar a guardar desde stage 3
        verbose:      int  = 1,
    ):
        super().__init__(verbose)
        self.league       = league
        self.scheduler    = scheduler
        self.every        = every
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.min_stage    = min_stage
        self._last_save   = 0

    def _on_step(self) -> bool:
        if self.scheduler.current_stage < self.min_stage:
            return True
        if self.num_timesteps - self._last_save < self.every:
            return True
        self._last_save = self.num_timesteps
        path = self.snapshot_dir / f"snap_t{self.num_timesteps}.zip"
        try:
            self.model.save(str(path))
            label = f"stage{self.scheduler.current_stage}_t{self.num_timesteps//1000}k"
            entry = self.league.add(str(path), self.num_timesteps, label=label)
            if self.verbose:
                print(f"\n[league] snapshot id={entry.snapshot_id} ({label}) "
                      f"saved → pool size {len(self.league)}")
        except Exception as ex:
            if self.verbose:
                print(f"\n[league] snapshot failed: {ex}")
        return True


# ── 7. LeagueResultCallback ──────────────────────────────────────

class LeagueResultCallback(BaseCallback):
    """
    Cuando una batalla termina y el oponente actual es un LeagueOpponent,
    actualiza el win-rate de ese snapshot en el league.

    El opponent_provider devuelve, dado un env_idx, el snapshot_id activo
    (-1 si no hay opp del league). Lo seteamos cuando rotamos opps.
    """

    def __init__(self, league, opponent_provider, verbose: int = 0):
        super().__init__(verbose)
        self.league = league
        self.opponent_provider = opponent_provider

    def _on_step(self) -> bool:
        infos = _as_sequence(self.locals.get("infos"))
        dones = _as_sequence(self.locals.get("dones"))
        for i, info in enumerate(infos):
            if i >= len(dones) or not dones[i]:
                continue
            sid = self.opponent_provider(i)
            if sid is None or sid < 0:
                continue
            won = bool(info.get("won", False))
            self.league.record_result(sid, won)
        return True
