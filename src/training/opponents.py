"""
src.training.opponents · Players para self-play.

  · LeagueOpponent — Player que carga un snapshot RL y ejecuta su política.
                     Mantiene LSTM state por battle_tag.
  · action_int_to_order — convierte el entero del action space en BattleOrder
                          de poke-env (intenta usar el helper interno y cae a
                          fallback manual si la versión de poke-env no lo expone).
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from poke_env.player import Player, RandomPlayer
from poke_env.battle.double_battle import DoubleBattle


# ── Action int → BattleOrder ─────────────────────────────────────

def action_int_to_order(action_int: int, slot_idx: int, battle: DoubleBattle):
    """
    Convierte un entero del action space (0..107) en BattleOrder para
    el slot dado. Imports defensivos porque la API de poke-env >= 0.8
    es algo movediza.

    Mapeo (Gen 9 dobles):
       -2     → default
       -1     → forfeit
        0     → no-op (pass)
        1–6   → switch al equipo[N-1]
        7–86  → move 1..4 con target -2..+2 (4 modes vacíos: mega/z/dyna/none)
        87–106 → move 1..4 con target -2..+2 + Tera
    """
    try:
        from poke_env.player.battle_order import (
            BattleOrder, DefaultBattleOrder
        )
    except ImportError:
        try:
            from poke_env.environment.battle_order import (
                BattleOrder, DefaultBattleOrder
            )
        except ImportError:
            BattleOrder = DefaultBattleOrder = None  # type: ignore

    a = int(action_int)

    # Pass / default / forfeit
    if a == 0 or a == -2:
        return DefaultBattleOrder() if DefaultBattleOrder else None
    if a == -1:
        # forfeit: no es algo que queramos hacer en self-play
        return DefaultBattleOrder() if DefaultBattleOrder else None

    # Switch
    if 1 <= a <= 6:
        team = list(battle.team.values())
        idx = a - 1
        if idx < 0 or idx >= len(team):
            return DefaultBattleOrder() if DefaultBattleOrder else None
        target_poke = team[idx]
        if target_poke.fainted or target_poke.active:
            return DefaultBattleOrder() if DefaultBattleOrder else None
        if BattleOrder is None:
            return None
        return BattleOrder(target_poke)

    # Move
    if 7 <= a <= 106:
        rel = a - 7
        block    = rel // 20    # 0=base, 1=mega, 2=z, 3=dyna, 4=tera
        within   = rel % 20
        move_idx = within // 5
        target   = (within % 5) - 2
        tera     = (block == 4)

        actives = battle.active_pokemon
        if slot_idx >= len(actives) or actives[slot_idx] is None:
            return DefaultBattleOrder() if DefaultBattleOrder else None
        active = actives[slot_idx]
        moves_list = list(active.moves.values())
        if move_idx >= len(moves_list):
            return DefaultBattleOrder() if DefaultBattleOrder else None
        move = moves_list[move_idx]
        if BattleOrder is None:
            return None
        try:
            return BattleOrder(move, move_target=target, terastallize=tera)
        except TypeError:
            # Versión más vieja: signaturas distintas
            try:
                return BattleOrder(move, terastallize=tera)
            except TypeError:
                return BattleOrder(move)

    return DefaultBattleOrder() if DefaultBattleOrder else None


def actions_to_double_order(actions: np.ndarray, battle: DoubleBattle):
    """Construye la DoubleBattleOrder a partir del array MultiDiscrete([107,107])."""
    try:
        from poke_env.player.battle_order import DoubleBattleOrder
    except ImportError:
        try:
            from poke_env.environment.battle_order import DoubleBattleOrder
        except ImportError:
            DoubleBattleOrder = None    # type: ignore

    a1, a2 = int(actions[0]), int(actions[1])
    o1 = action_int_to_order(a1, slot_idx=0, battle=battle)
    o2 = action_int_to_order(a2, slot_idx=1, battle=battle)

    if DoubleBattleOrder is not None:
        try:
            return DoubleBattleOrder(o1, o2)
        except Exception:
            pass
    # Fallback: poke-env algunas versiones aceptan list
    return [o1, o2]


# ── LeagueOpponent ───────────────────────────────────────────────

class LeagueOpponent(Player):
    """
    Player que ejecuta la política de un snapshot RL guardado.

    Args:
        model:        modelo cargado (RecurrentPPO o MaskablePPO)
        encoder_env:  un VGCEnv "proxy" (no listening) cuyo embed_battle()
                      reusamos para encodear el estado igual que el trainee
        algorithm:    "recurrent_ppo" | "maskable_ppo"
        snapshot_id:  id del snapshot en el league (para record_result)
        deterministic: True para play deterministico
    """

    def __init__(
        self,
        model,
        encoder_env,           # VGCEnv usado solo para .embed_battle()
        algorithm:    str  = "recurrent_ppo",
        snapshot_id:  int  = -1,
        deterministic: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.encoder_env = encoder_env
        self.algorithm = algorithm
        self.snapshot_id = snapshot_id
        self.deterministic = deterministic
        self.lstm_states: dict[str, object] = {}

    def choose_move(self, battle: DoubleBattle):
        # 1) Encodear igual que el trainee
        try:
            obs = self.encoder_env.embed_battle(battle)
        except Exception:
            return self.choose_random_move(battle)

        obs = np.asarray(obs, dtype=np.float32)[None, :]   # batch dim

        # 2) Predict (LSTM mantiene memoria por battle_tag)
        try:
            if self.algorithm == "recurrent_ppo":
                state = self.lstm_states.get(battle.battle_tag)
                ep_start = np.array([state is None], dtype=bool)
                action, new_state = self.model.predict(
                    obs,
                    state         = state,
                    episode_start = ep_start,
                    deterministic = self.deterministic,
                )
                self.lstm_states[battle.battle_tag] = new_state
            else:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
        except Exception:
            return self.choose_random_move(battle)

        # 3) Convertir a BattleOrder
        try:
            return actions_to_double_order(action[0], battle)
        except Exception:
            return self.choose_random_move(battle)

    def reset_lstm_state(self, battle_tag: str | None = None) -> None:
        if battle_tag is None:
            self.lstm_states.clear()
        else:
            self.lstm_states.pop(battle_tag, None)
