"""
vgc_env.py
─────────────────────────────────────────────────────────────────
Environment VGC (Gen 9 Dobles) usando poke-env's DoublesEnv.

poke-env >= 0.8 usa la API de PettingZoo (paralela, dos agentes).
Este archivo subclasea DoublesEnv e implementa los dos métodos
abstractos requeridos:

  calc_reward(battle)  → float
  embed_battle(battle) → np.ndarray

Para entrenar con stable-baselines3 (que espera Gymnasium single-agent),
se usa SingleAgentWrapper, que convierte el env paralelo en uno estándar
donde el agente1 es el que entrenamos y el agente2 es un oponente bot.

Flujo:
  env   = VGCEnv(team=team_str, battle_format="gen9vgc2025regg")
  opp   = RandomPlayer(battle_format="gen9vgc2025regg", team=team_str)
  gym_env = SingleAgentWrapper(env, opp)
  # gym_env ya tiene obs = {"observation": array, "action_mask": array}
  # Usar FlatObsWrapper para PPO estándar de SB3

Espacio de acciones (DoublesEnv, Gen 9):
  MultiDiscrete([107, 107])  — una acción por cada Pokémon activo
  Mapeo de cada entero:
    -2     → default (poke-env elige)
    -1     → rendirse
     0     → pass
     1–6   → switch al Pokémon de equipo N
     7–11  → move 1, target -2..+2
    12–16  → move 2, target -2..+2
    17–21  → move 3, target -2..+2
    22–26  → move 4, target -2..+2
    27–46  → + mega evolve
    47–66  → + z-move
    67–86  → + dynamax
    87–106 → + Terastal
"""

from __future__ import annotations

import numpy as np
from typing import Optional
from pathlib import Path
from gymnasium import spaces

from poke_env.environment import DoublesEnv
from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.weather import Weather
from poke_env.battle.field import Field
from poke_env.battle.pokemon import Pokemon as PokemonObj

from src.state_encoder import StateEncoder
from src.damage_calc import BattleConditions
from src.utils import load_all_data, parse_team, calc_all_stats, get_pokemon

# ── Pesos de la función de recompensa ─────────────────────────────
W_DMG_DEALT  =  1.0   # por cada 1% de HP infligido
W_DMG_TAKEN  = -0.8   # por cada 1% de HP recibido
W_KO         =  3.0   # por cada KO infligido
W_KO_TAKEN   = -2.5   # por cada KO recibido
W_WIN        = 15.0   # recompensa / penalidad final

# ── Mapeos de clima y terreno ──────────────────────────────────────
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
    Environment VGC para RL, listo para stable-baselines3.

    Hereda de DoublesEnv (poke-env), que implementa la API PettingZoo
    para batallas dobles de Gen 9.

    Para entrenamiento con SB3:
        from poke_env.environment import SingleAgentWrapper
        from poke_env import RandomPlayer

        env = VGCEnv(team=team_str, battle_format="gen9vgc2025regg")
        opp = RandomPlayer(battle_format="gen9vgc2025regg", team=team_str)
        gym_env = FlatObsWrapper(SingleAgentWrapper(env, opp))
        model   = PPO("MlpPolicy", gym_env, ...)
    """

    def __init__(
        self,
        team_path: str | Path = "team.txt",
        **kwargs,
    ):
        # ── Cargar datos y equipo ─────────────────────────────────
        self.data      = load_all_data()
        self.team_list = parse_team(team_path)

        # Precalcular stats y tipos del equipo propio
        self._team_stats: dict[str, dict] = {}
        self._team_types: dict[str, list] = {}
        for p in self.team_list:
            stats = calc_all_stats(p, self.data["pokemon"])
            poke_info = get_pokemon(p["name"], self.data["pokemon"])
            self._team_stats[p["name"]] = stats
            self._team_types[p["name"]] = poke_info["types"] if poke_info else []

        # Leer el equipo como string en formato Pokepaste
        tp = Path(team_path)
        with open(tp, encoding="utf-8") as f:
            team_str = f.read()

        # ── Inicializar DoublesEnv ────────────────────────────────
        # Los kwargs que no especifiquemos toman defaults de DoublesEnv:
        #   battle_format = "gen8randombattle" → se sobreescribe con kwarg
        #   server_configuration = LocalhostServerConfiguration
        #   start_listening = True
        super().__init__(team=team_str, **kwargs)

        # ── Encoder del estado ────────────────────────────────────
        self.encoder = StateEncoder(self.data["type_chart"], self.data["moves"])
        obs_size = self.encoder.get_obs_shape()[0]

        # ── Observation spaces (se setean por agente en DoublesEnv) ──
        # DoublesEnv.__setattr__ los envuelve automáticamente en
        # Dict({"observation": Box, "action_mask": Box})
        self.observation_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)
            for agent in self.possible_agents
        }

        # ── Estado interno para el reward ─────────────────────────
        self._prev_own_hp:   dict[str, list] = {}
        self._prev_rival_hp: dict[str, list] = {}
        self._prev_own_ko:   dict[str, int]  = {}
        self._prev_rival_ko: dict[str, int]  = {}

    # ── Métodos abstractos requeridos por DoublesEnv ──────────────

    def calc_reward(self, battle: DoubleBattle) -> float:
        """
        Calcula el reward del turno.

        poke-env llama a este método desde step() y reset().
        """
        tag = battle.battle_tag

        # Inicializar estado si es la primera vez que vemos esta batalla
        if tag not in self._prev_own_hp:
            self._prev_own_hp[tag]   = [1.0, 1.0]
            self._prev_rival_hp[tag] = [1.0, 1.0]
            self._prev_own_ko[tag]   = 0
            self._prev_rival_ko[tag] = 0

        reward = 0.0

        # ── HP actual de los activos ──────────────────────────────
        own_active   = battle.active_pokemon            # List[Optional[Pokemon]]
        rival_active = battle.opponent_active_pokemon   # List[Optional[Pokemon]]

        curr_own_hp   = [p.current_hp_fraction if p else 0.0 for p in own_active]
        curr_rival_hp = [p.current_hp_fraction if p else 0.0 for p in rival_active]

        # Padding a longitud 2
        while len(curr_own_hp)   < 2: curr_own_hp.append(0.0)
        while len(curr_rival_hp) < 2: curr_rival_hp.append(0.0)

        # ── Daño infligido / recibido ─────────────────────────────
        prev_own   = self._prev_own_hp[tag]
        prev_rival = self._prev_rival_hp[tag]

        for i in range(2):
            dmg_dealt = prev_rival[i] - curr_rival_hp[i]
            dmg_taken = prev_own[i]   - curr_own_hp[i]
            if dmg_dealt > 0:
                reward += dmg_dealt * 100 * W_DMG_DEALT
            if dmg_taken > 0:
                reward += dmg_taken * 100 * W_DMG_TAKEN   # W_DMG_TAKEN es negativo

        # ── KOs ───────────────────────────────────────────────────
        curr_rival_ko = sum(1 for p in battle.opponent_team.values() if p.fainted)
        curr_own_ko   = sum(1 for p in battle.team.values()          if p.fainted)

        new_rival_ko = curr_rival_ko - self._prev_rival_ko[tag]
        new_own_ko   = curr_own_ko   - self._prev_own_ko[tag]

        reward += new_rival_ko * W_KO
        reward += new_own_ko   * W_KO_TAKEN   # negativo

        # ── Victoria / derrota ────────────────────────────────────
        if battle.finished:
            if battle.won:
                reward += W_WIN
            else:
                reward -= W_WIN
            # Limpiar estado de esta batalla
            self._prev_own_hp.pop(tag, None)
            self._prev_rival_hp.pop(tag, None)
            self._prev_own_ko.pop(tag, None)
            self._prev_rival_ko.pop(tag, None)
        else:
            self._prev_own_hp[tag]   = list(curr_own_hp)
            self._prev_rival_hp[tag] = list(curr_rival_hp)
            self._prev_rival_ko[tag] = curr_rival_ko
            self._prev_own_ko[tag]   = curr_own_ko

        return reward

    def embed_battle(self, battle: DoubleBattle) -> np.ndarray:
        """
        Convierte el estado actual de la batalla en el vector numpy
        que entiende la red neuronal.

        poke-env llama a este método para construir la observación.
        Retorna solo el array puro (el env lo envuelve en el dict
        {"observation": ..., "action_mask": ...} automáticamente).
        """
        # ── Propios en campo ─────────────────────────────────────
        own_field = []
        for poke in battle.active_pokemon:
            if poke is None:
                continue
            own_field.append(self._encode_poke(poke, is_ally=True))

        # ── Rivales en campo ──────────────────────────────────────
        rival_field = []
        for poke in battle.opponent_active_pokemon:
            if poke is None:
                continue
            rival_field.append(self._encode_poke(poke, is_ally=False))

        # ── Banqueados propios ────────────────────────────────────
        active_species = {p.species for p in battle.active_pokemon if p}
        benched = []
        for poke in battle.team.values():
            if poke.species not in active_species and not poke.fainted:
                benched.append({
                    "hp_pct": poke.current_hp_fraction,
                    "types":  [t.name.lower() for t in poke.types if t],
                    "status": poke.status.name.lower() if poke.status else None,
                })

        # ── Condiciones del campo ─────────────────────────────────
        # battle.weather devuelve Dict[Weather, turn] en poke-env nuevo
        weather_enum = next(iter(battle.weather), None)
        weather = WEATHER_MAP.get(weather_enum, "none") if weather_enum else "none"

        terrain = "none"
        for field_enum in battle.fields:
            t = TERRAIN_MAP.get(field_enum)
            if t:
                terrain = t
                break

        trick_room = Field.TRICK_ROOM in battle.fields

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

        # Stats: reales si es nuestro, base stats estimados si es rival
        if is_ally and poke.species in self._team_stats:
            stats = self._team_stats[poke.species]
        else:
            poke_info = get_pokemon(poke.species, self.data["pokemon"])
            stats = dict(poke_info["stats"]) if poke_info else {}

        # Modificadores de stat
        stat_mods = {}
        if hasattr(poke, "boosts") and poke.boosts:
            boost_map = {
                "atk": "attack", "def": "defense",
                "spa": "special-attack", "spd": "special-defense",
                "spe": "speed",
            }
            for abbr, full in boost_map.items():
                val = getattr(poke.boosts, abbr, 0)
                if val:
                    stat_mods[full] = val

        # Movimientos conocidos
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
            "tera_available": not poke.terastallized,
            "tera_type":      None,
            "item":           poke.item if hasattr(poke, "item") else None,
        }


# ── Wrapper para SB3 (extrae la observación del dict) ─────────────

import gymnasium as gym

class FlatObsWrapper(gym.Wrapper):
    """
    Convierte el env de SingleAgentWrapper (obs = Dict con
    "observation" y "action_mask") en un env con obs = Box puro.

    Necesario para usar PPO estándar de stable-baselines3.
    Si querés usar MaskablePPO (sb3-contrib) podés prescindir de
    este wrapper y pasar el env de SingleAgentWrapper directamente.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Sobreescribir el observation_space con solo el Box interno
        self.observation_space = env.observation_space["observation"]

    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        return obs_dict["observation"], reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        return obs_dict["observation"], info


# ── Entry point de prueba (sin Showdown) ─────────────────────────

if __name__ == "__main__":
    from src.state_encoder import StateEncoder
    from src.utils import load_all_data

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
