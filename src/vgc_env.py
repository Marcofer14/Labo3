"""
vgc_env.py
─────────────────────────────────────────────────────────────────
Environment VGC (Gen 9 Dobles) usando poke-env's DoublesEnv.

Hereda de DoublesEnv (PettingZoo paralelo, dos agentes) e implementa:
  calc_reward(battle)  → float
  embed_battle(battle) → np.ndarray

Wave 1 — Curriculum learning:
  • Reward delegado a src.rewards.RewardCalculator (4 capas A/B/C/D)
  • step() interceptado → guarda última acción por agente para que
    los reward modules puedan razonar sobre "qué hicimos"
  • Config de reward inyectable en runtime (curriculum scheduler)

Para entrenar con stable-baselines3 / sb3-contrib:
    env     = VGCEnv(team_path="team.txt", battle_format=...)
    opp     = RandomPlayer(...)
    gym_env = SingleAgentWrapper(env, opp)         # PettingZoo → Gymnasium
    # Para PPO clásico: FlatObsWrapper(gym_env)
    # Para MaskablePPO: MaskedFlatObsWrapper(gym_env)
"""

from __future__ import annotations

import numpy as np
from typing import Optional
from pathlib import Path
from gymnasium import spaces
import gymnasium as gym

from poke_env.environment import DoublesEnv
from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.weather       import Weather
from poke_env.battle.field         import Field
from poke_env.battle.pokemon       import Pokemon as PokemonObj

from src.state_encoder import StateEncoder
from src.utils         import load_all_data, parse_team, calc_all_stats, get_pokemon

from src.rewards               import RewardConfig, RewardCalculator
from src.rewards.action_decoder import decode_action


# ── Mapeos de clima y terreno (encoder usa strings) ─────────────────
WEATHER_MAP = {
    Weather.RAINDANCE:     "rain",
    Weather.PRIMORDIALSEA: "rain",
    Weather.SUNNYDAY:      "sun",
    Weather.DESOLATELAND:  "sun",
    Weather.SANDSTORM:     "sandstorm",
    Weather.SNOWSCAPE:     "snow",
    Weather.HAIL:          "snow",
}
TERRAIN_MAP = {
    Field.GRASSY_TERRAIN:   "grassy",
    Field.ELECTRIC_TERRAIN: "electric",
    Field.PSYCHIC_TERRAIN:  "psychic",
    Field.MISTY_TERRAIN:    "misty",
}


class VGCEnv(DoublesEnv):
    """
    Environment VGC para RL, listo para SB3 / sb3-contrib.

    Args:
        team_path:      ruta al .txt en formato Pokepaste
        reward_config:  RewardConfig (default: stage_1)
        kwargs:         se pasan a DoublesEnv (battle_format, server_configuration, ...)
    """

    def __init__(
        self,
        team_path: str | Path = "team.txt",
        reward_config: Optional[RewardConfig] = None,
        **kwargs,
    ):
        # ── Datos y equipo ───────────────────────────────────────
        self.data      = load_all_data()
        self.team_list = parse_team(team_path)

        self._team_stats: dict[str, dict] = {}
        self._team_types: dict[str, list] = {}
        for p in self.team_list:
            stats = calc_all_stats(p, self.data["pokemon"])
            poke_info = get_pokemon(p["name"], self.data["pokemon"])
            self._team_stats[p["name"]] = stats
            self._team_types[p["name"]] = poke_info["types"] if poke_info else []

        with open(Path(team_path), encoding="utf-8") as f:
            team_str = f.read()

        super().__init__(team=team_str, **kwargs)

        # ── State encoder ────────────────────────────────────────
        self.encoder = StateEncoder(self.data["type_chart"], self.data["moves"])
        obs_size = self.encoder.get_obs_shape()[0]

        self.observation_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)
            for agent in self.possible_agents
        }

        # ── Reward calculator (4 capas) ──────────────────────────
        self.reward_config = reward_config or RewardConfig.stage_1()
        self.reward_calc   = RewardCalculator(
            config     = self.reward_config,
            data       = self.data,
            team_stats = self._team_stats,
        )

        # ── Acciones último step por agente (para reward modules) ──
        # mapping: agent_name -> tuple[DecodedAction, DecodedAction]
        self._last_actions: dict[str, tuple] = {}

        # Buffer del breakdown del último compute (lo expone via info)
        self._last_breakdown: dict[str, dict[str, float]] = {}

        # Flag de victoria de la última batalla finalizada (por tag).
        # WonInfoWrapper lo lee y lo pone en info["won"].
        self._last_battle_won: dict[str, bool] = {}
        # Última batalla finalizada (para que los wrappers la encuentren)
        self._last_finished_tag: Optional[str] = None

    # ── Curriculum API ───────────────────────────────────────────

    def set_reward_config(self, cfg: RewardConfig) -> None:
        """Cambia la config de reward en caliente (curriculum scheduler)."""
        self.reward_config = cfg
        self.reward_calc.set_config(cfg)

    @staticmethod
    def action_to_order(action, battle: DoubleBattle, fake: bool = False, strict: bool = True):
        """
        Conversor tolerante para PPO/RecurrentPPO.

        La policy puede proponer acciones inválidas antes de aprender el action
        space. El conversor base de poke-env lanza ValueError en esos casos y
        corta el entrenamiento; acá caemos a default/pass para que la penalización
        venga por reward en vez de romper el rollout.
        """
        try:
            return DoublesEnv.action_to_order(action, battle, fake=fake, strict=strict)
        except Exception:
            from src.training.opponents import actions_to_double_order

            return actions_to_double_order(action, battle)

    # ── Action interception ──────────────────────────────────────

    def step(self, actions):
        """
        Interceptamos las acciones antes de pasarlas al env padre para que
        los reward modules sepan qué eligió cada agente.

        `actions` es {agent_name: np.ndarray([a1, a2])}.
        """
        try:
            for agent, raw in (actions or {}).items():
                a1, a2 = int(raw[0]), int(raw[1])
                self._last_actions[agent] = (decode_action(a1), decode_action(a2))
        except Exception:
            # Nunca rompemos el step por un decoding fallido
            pass
        try:
            return super().step(actions)
        except AssertionError:
            if not self._has_finished_battle():
                raise
            return self._finished_step_result()

    def reset(self, *args, **kwargs):
        self._last_actions.clear()
        return super().reset(*args, **kwargs)

    def _has_finished_battle(self) -> bool:
        return any(
            bool(getattr(battle, "finished", False))
            for battle in (getattr(self, "battle1", None), getattr(self, "battle2", None))
        )

    def _obs_for_battle(self, battle: Optional[DoubleBattle]) -> dict:
        if battle is None:
            obs = np.zeros(self.observation_spaces[self.possible_agents[0]].shape, dtype=np.float32)
            mask = np.array([], dtype=bool)
        else:
            obs = self.embed_battle(battle)
            try:
                mask = np.array(self.get_action_mask(battle))
            except Exception:
                mask = np.array([], dtype=bool)
        return {"observation": obs, "action_mask": mask}

    def _finished_step_result(self):
        battle1 = getattr(self, "battle1", None)
        battle2 = getattr(self, "battle2", None)
        agent1, agent2 = self.possible_agents[0], self.possible_agents[1]

        observations = {
            agent1: self._obs_for_battle(battle1),
            agent2: self._obs_for_battle(battle2),
        }
        rewards = {
            agent1: self.calc_reward(battle1) if battle1 is not None else 0.0,
            agent2: self.calc_reward(battle2) if battle2 is not None else 0.0,
        }

        term1, trunc1 = self.calc_term_trunc(battle1) if battle1 is not None else (True, False)
        term2, trunc2 = self.calc_term_trunc(battle2) if battle2 is not None else (True, False)
        if self._has_finished_battle():
            term1 = term1 or not trunc1
            term2 = term2 or not trunc2

        terminated = {agent1: bool(term1), agent2: bool(term2)}
        truncated = {agent1: bool(trunc1), agent2: bool(trunc2)}
        infos = self.get_additional_info()

        self.agents = []
        return observations, rewards, terminated, truncated, infos

    # ── Métodos abstractos requeridos por DoublesEnv ─────────────

    def calc_reward(self, battle: DoubleBattle) -> float:
        """
        Delegado al RewardCalculator (capas A + B + C + D según config).

        El calculator se encarga de: snapshot, diff con turno previo,
        cleanup al terminar la batalla.
        """
        # Resolver agent name → buscamos el agente cuyo battle es este
        last = None
        for agent_name, pair in self._last_actions.items():
            try:
                # En DoublesEnv hay un agent_to_battle dict-like; fallback a None
                if hasattr(self, "agent1") and getattr(self.agent1, "battle", None) is battle:
                    last = pair; break
            except Exception:
                pass
        # Fallback: si solo hay un par de acciones almacenadas, usarlo
        if last is None and len(self._last_actions) == 1:
            last = next(iter(self._last_actions.values()))

        reward, breakdown = self.reward_calc.compute(battle, last_actions=last)
        # Guardamos el breakdown para que los callbacks lo lean
        self._last_breakdown[battle.battle_tag] = breakdown
        # Si la batalla terminó, registramos won + flagueamos la batalla
        if battle.finished:
            self._last_battle_won[battle.battle_tag] = bool(battle.won)
            self._last_finished_tag = battle.battle_tag
        return reward

    def embed_battle(self, battle: DoubleBattle) -> np.ndarray:
        """Vector de observación normalizado [0, 1] para la red neuronal."""
        own_field = []
        for poke in battle.active_pokemon:
            if poke is None:
                continue
            own_field.append(self._encode_poke(poke, is_ally=True))

        rival_field = []
        for poke in battle.opponent_active_pokemon:
            if poke is None:
                continue
            rival_field.append(self._encode_poke(poke, is_ally=False))

        active_species = {p.species for p in battle.active_pokemon if p}
        benched = []
        for poke in battle.team.values():
            if poke.species not in active_species and not poke.fainted:
                benched.append({
                    "hp_pct": poke.current_hp_fraction,
                    "types":  [t.name.lower() for t in poke.types if t],
                    "status": poke.status.name.lower() if poke.status else None,
                })

        weather_enum = next(iter(battle.weather), None) if battle.weather else None
        weather = WEATHER_MAP.get(weather_enum, "none") if weather_enum else "none"

        terrain = "none"
        for field_enum in (battle.fields or {}):
            t = TERRAIN_MAP.get(field_enum)
            if t:
                terrain = t
                break

        trick_room = Field.TRICK_ROOM in (battle.fields or {})

        conditions = {
            "weather":    weather,
            "terrain":    terrain,
            "trick_room": trick_room,
            "turn":       battle.turn,
        }

        return self.encoder.encode_manual(own_field, rival_field, benched, conditions)

    # ── Helpers internos ──────────────────────────────────────────

    def _encode_poke(self, poke: PokemonObj, is_ally: bool) -> dict:
        """Convierte un Pokemon de poke-env al formato que espera StateEncoder."""
        types = [t.name.lower() for t in poke.types if t]

        if is_ally and poke.species in self._team_stats:
            stats = self._team_stats[poke.species]
        else:
            poke_info = get_pokemon(poke.species, self.data["pokemon"])
            stats = dict(poke_info["stats"]) if poke_info else {}

        stat_mods = {}
        if hasattr(poke, "boosts") and poke.boosts:
            boost_map = {
                "atk": "attack", "def": "defense",
                "spa": "special-attack", "spd": "special-defense",
                "spe": "speed",
            }
            for abbr, full in boost_map.items():
                val = getattr(poke.boosts, abbr, 0) if hasattr(poke.boosts, abbr) else (
                    poke.boosts.get(abbr, 0) if isinstance(poke.boosts, dict) else 0
                )
                if val:
                    stat_mods[full] = val

        moves_list = []
        for move in poke.moves.values():
            move_dict = self.data["moves"].get(move.id)
            if move_dict:
                move_dict = dict(move_dict)
                move_dict["pp_left"] = move.current_pp
                moves_list.append(move_dict)

        return {
            "hp_pct":         poke.current_hp_fraction,
            "types":          types,
            "stats":          stats,
            "stat_mods":      stat_mods,
            "status":         poke.status.name.lower() if poke.status else None,
            "moves":          [m.get("name", "") for m in moves_list],
            "tera_available": not getattr(poke, "terastallized", getattr(poke, "_terastallized", False)),
            "tera_type":      None,
            "item":           poke.item if hasattr(poke, "item") else None,
        }


# ── Wrappers para SB3 (extracción de obs del dict) ────────────────

def _find_vgc_env(env: gym.Env) -> Optional["VGCEnv"]:
    """Camina la pila de wrappers buscando el VGCEnv."""
    cur = env
    while cur is not None:
        if isinstance(cur, VGCEnv):
            return cur
        cur = getattr(cur, "env", None)
    return None


class FlatObsWrapper(gym.Wrapper):
    """
    PPO clásico: extrae solo "observation" del dict y propaga
    `won` + `reward_breakdown` a info para los callbacks.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = env.observation_space["observation"]
        self._vgc = _find_vgc_env(env)

    def _enrich_info(self, info, terminated, truncated):
        if not (terminated or truncated):
            return info
        info = dict(info) if info else {}
        if self._vgc is not None:
            tag = self._vgc._last_finished_tag
            if tag and tag in self._vgc._last_battle_won:
                info["won"] = self._vgc._last_battle_won.pop(tag)
            if tag and tag in self._vgc._last_breakdown:
                info["reward_breakdown"] = self._vgc._last_breakdown.pop(tag)
        return info

    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        info = self._enrich_info(info, terminated, truncated)
        return obs_dict["observation"], reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        return obs_dict["observation"], info


class MaskedFlatObsWrapper(gym.Wrapper):
    """
    Para MaskablePPO (sb3-contrib): observation queda como Box plano,
    pero exponemos action_masks() para que la policy lo recoja.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = env.observation_space["observation"]
        self._last_mask: Optional[np.ndarray] = None

    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        self._last_mask = obs_dict.get("action_mask")
        return obs_dict["observation"], reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        self._last_mask = obs_dict.get("action_mask")
        return obs_dict["observation"], info

    def action_masks(self) -> np.ndarray:
        """Llamado por MaskablePPO antes de samplear."""
        if self._last_mask is None:
            # Sin máscara conocida → todas válidas
            n = self.action_space.nvec.sum() if hasattr(self.action_space, "nvec") else self.action_space.n
            return np.ones(int(n), dtype=bool)
        return self._last_mask.astype(bool)


# ── Entry point de prueba (sin Showdown) ─────────────────────────

if __name__ == "__main__":
    from src.state_encoder import StateEncoder
    from src.utils         import load_all_data

    data    = load_all_data()
    encoder = StateEncoder(data["type_chart"], data["moves"])
    obs     = encoder.encode_manual([], [], [], {})

    print(f"VGCEnv — verificación del StateEncoder:")
    print(f"  obs_shape: {obs.shape}")
    print(f"  min={obs.min():.4f}  max={obs.max():.4f}")
    print(f"  ✓ StateEncoder OK")
    print(f"")
    print(f"  DoublesEnv action_space_size (Gen 9): {DoublesEnv.get_action_space_size(9)}")
    print(f"  → MultiDiscrete([{DoublesEnv.get_action_space_size(9)}, {DoublesEnv.get_action_space_size(9)}])")
    print(f"  ✓ VGCEnv listo para conectar a Showdown y entrenar.")
