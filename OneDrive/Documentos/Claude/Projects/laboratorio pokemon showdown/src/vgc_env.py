"""
vgc_env.py
─────────────────────────────────────────────────────────────────
Environment Gymnasium para VGC (Gen 9 Dobles) usando poke-env.

Este archivo envuelve la interfaz de poke-env con Gymnasium para
que el agente de RL (PPO, DQN, etc.) pueda interactuar con él.

Flujo de una batalla:
  env.reset() → observación inicial
  env.step(action) → nueva observación, reward, done, info
  ... (loop hasta que done == True)

Espacio de acciones (action space):
  Discreto. Cada entero mapea a una acción concreta:
    0–3   → usar move 0–3 con primer Pokémon activo
    4–7   → usar move 0–3 con segundo Pokémon activo
    8–11  → usar move 0–3 + Terastal con primer Pokémon
    12    → switch primer Pokémon al banqueado 0
    13    → switch primer Pokémon al banqueado 1
    14    → switch segundo Pokémon al banqueado 0
    15    → switch segundo Pokémon al banqueado 1

  Total: 16 acciones discretas

Función de recompensa:
  Se acumula por turno y al final de la batalla:
  +  daño % infligido × W_DMG_DEALT
  -  daño % recibido  × W_DMG_TAKEN
  +  KO infligido     × W_KO
  -  KO recibido      × W_KO_TAKEN
  +  movimiento SE    × W_SUPER_EFF
  +  acción defensiva exitosa × W_DEFENSIVE
  +  victoria final   × W_WIN
  -  derrota final    × W_WIN
"""

import numpy as np
from gymnasium import spaces
import asyncio
from typing import Optional

try:
    from poke_env.environment import Battle, Pokemon, Move
    from poke_env.player import Gen9EnvSinglePlayer
    POKE_ENV_AVAILABLE = True
except ImportError:
    POKE_ENV_AVAILABLE = False
    print("⚠ poke-env no instalado. Instalar con: pip install poke-env")

from src.state_encoder import StateEncoder
from src.damage_calc import calc_damage, BattleConditions
from src.utils import (
    load_all_data, parse_team, calc_all_stats,
    get_move, get_pokemon, get_effectiveness
)
from pathlib import Path

# ── Pesos de la función de recompensa ─────────────────────────────
# Estos valores son los que habrá que tunear durante el entrenamiento.
W_DMG_DEALT  =  1.0   # por cada 1% de HP infligido
W_DMG_TAKEN  = -0.8   # por cada 1% de HP recibido
W_KO         =  3.0   # por cada KO infligido
W_KO_TAKEN   = -2.5   # por cada KO recibido
W_SUPER_EFF  =  0.5   # bonus por usar movimiento superefectivo (x2 o x4)
W_DEFENSIVE  =  0.3   # bonus por acción defensiva exitosa (Protect que bloqueó, switch útil)
W_WIN        = 15.0   # recompensa/penalidad final por ganar/perder

# ── Número de acciones discretas ──────────────────────────────────
N_ACTIONS = 16


class VGCEnv(Gen9EnvSinglePlayer if POKE_ENV_AVAILABLE else object):
    """
    Environment VGC para RL.

    Hereda de Gen9EnvSinglePlayer de poke-env y lo adapta para:
      • Formato dobles (VGC)
      • Vector de observación del StateEncoder
      • Función de recompensa diseñada para VGC
      • Acción discreta con 16 opciones

    Uso básico:
        env = VGCEnv(team_path="team.txt")
        obs, info = env.reset()
        action = agent.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
    """

    def __init__(
        self,
        team_path:  str | Path = "team.txt",
        data_dir:   Optional[str] = None,
        **kwargs,
    ):
        # Cargar datos y equipo
        self.data       = load_all_data()
        self.team       = parse_team(team_path)
        self.encoder    = StateEncoder(self.data["type_chart"], self.data["moves"])

        # Precalcular stats del equipo propio
        self.team_stats: dict[str, dict] = {}
        self.team_types: dict[str, list] = {}
        for p in self.team:
            stats = calc_all_stats(p, self.data["pokemon"])
            poke_info = get_pokemon(p["name"], self.data["pokemon"])
            self.team_stats[p["name"]] = stats
            self.team_types[p["name"]] = poke_info["types"] if poke_info else []

        # Obtener tamaño del observation space
        dummy_obs = self.encoder.encode_manual([], [], [], {})
        obs_size  = dummy_obs.shape[0]

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low   = 0.0,
            high  = 1.0,
            shape = (obs_size,),
            dtype = np.float32,
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Estado interno del reward
        self._prev_own_hp:   list[float] = [1.0, 1.0]
        self._prev_rival_hp: list[float] = [1.0, 1.0]
        self._prev_own_ko:   int = 0
        self._prev_rival_ko: int = 0

        if POKE_ENV_AVAILABLE:
            super().__init__(**kwargs)

    # ── Gymnasium API ─────────────────────────────────────────────

    def calc_reward(self, last_battle: "Battle") -> float:
        """
        Calcula el reward del turno actual comparando el estado
        anterior con el estado actual.

        poke-env llama a este método automáticamente en cada step().
        """
        reward = 0.0

        # ── Sacar HP actuales de los activos ──────────────────────
        own_active   = list(last_battle.active_pokemon.values())
        rival_active = list(last_battle.opponent_active_pokemon.values())

        curr_own_hp   = [p.current_hp_fraction for p in own_active   if p]
        curr_rival_hp = [p.current_hp_fraction for p in rival_active if p]

        # Padding si hay menos de 2 Pokémon en campo
        while len(curr_own_hp)   < 2: curr_own_hp.append(0.0)
        while len(curr_rival_hp) < 2: curr_rival_hp.append(0.0)

        # ── Daño infligido y recibido ──────────────────────────────
        for i in range(2):
            dmg_dealt = self._prev_rival_hp[i] - curr_rival_hp[i]
            dmg_taken = self._prev_own_hp[i]   - curr_own_hp[i]

            if dmg_dealt > 0:
                reward += dmg_dealt * 100 * W_DMG_DEALT
                # Bonus si fue superefectivo (necesita info del último movimiento)
                # TODO: extraer tipo del último movimiento usado para verificar SE
            if dmg_taken > 0:
                reward += dmg_taken * 100 * W_DMG_TAKEN  # W_DMG_TAKEN es negativo

        # ── KOs ───────────────────────────────────────────────────
        curr_rival_ko = sum(1 for p in last_battle.opponent_team.values() if p.fainted)
        curr_own_ko   = sum(1 for p in last_battle.team.values()          if p.fainted)

        new_rival_ko = curr_rival_ko - self._prev_rival_ko
        new_own_ko   = curr_own_ko   - self._prev_own_ko

        reward += new_rival_ko * W_KO
        reward += new_own_ko   * W_KO_TAKEN  # W_KO_TAKEN es negativo

        # ── Victoria / derrota final ──────────────────────────────
        if last_battle.finished:
            if last_battle.won:
                reward += W_WIN
            else:
                reward -= W_WIN

        # ── Actualizar estado previo ──────────────────────────────
        self._prev_own_hp   = list(curr_own_hp)
        self._prev_rival_hp = list(curr_rival_hp)
        self._prev_rival_ko = curr_rival_ko
        self._prev_own_ko   = curr_own_ko

        return reward

    def embed_battle(self, battle: "Battle") -> np.ndarray:
        """
        Convierte el estado actual de una batalla de poke-env
        al vector numpy que entiende la red neuronal.

        poke-env llama a este método para obtener la observación.
        """
        # ── Pokémon propios en campo ──────────────────────────────
        own_field = []
        for poke in battle.active_pokemon.values():
            if poke is None:
                continue
            own_field.append(self._encode_poke_from_battle(poke, is_ally=True))

        # ── Pokémon rivales en campo ──────────────────────────────
        rival_field = []
        for poke in battle.opponent_active_pokemon.values():
            if poke is None:
                continue
            rival_field.append(self._encode_poke_from_battle(poke, is_ally=False))

        # ── Banqueados propios ────────────────────────────────────
        benched = []
        active_names = {p.species for p in battle.active_pokemon.values() if p}
        for poke in battle.team.values():
            if poke.species not in active_names and not poke.fainted:
                benched.append({
                    "hp_pct": poke.current_hp_fraction,
                    "types":  [t.name.lower() for t in poke.types if t],
                    "status": poke.status.name.lower() if poke.status else None,
                })

        # ── Condiciones del campo ─────────────────────────────────
        weather = "none"
        if battle.weather:
            weather_map = {
                "RAINDANCE": "rain", "SUNNYDAY": "sun",
                "SANDSTORM": "sandstorm", "SNOW": "snow", "HAIL": "snow",
            }
            weather = weather_map.get(battle.weather.name, "none")

        terrain = "none"
        if battle.fields:
            terrain_map = {
                "GRASSY_TERRAIN":   "grassy",
                "ELECTRIC_TERRAIN": "electric",
                "PSYCHIC_TERRAIN":  "psychic",
                "MISTY_TERRAIN":    "misty",
            }
            for field in battle.fields:
                terrain = terrain_map.get(field.name, "none")
                if terrain != "none":
                    break

        trick_room = any(f.name == "TRICK_ROOM" for f in battle.fields)

        conditions = {
            "weather":    weather,
            "terrain":    terrain,
            "trick_room": trick_room,
            "turn":       battle.turn,
        }

        return self.encoder.encode_manual(own_field, rival_field, benched, conditions)

    def _encode_poke_from_battle(self, poke: "Pokemon", is_ally: bool) -> dict:
        """Extrae los datos de un Pokémon de poke-env y los formatea para el encoder."""
        types = [t.name.lower() for t in poke.types if t]

        # Stats: usar los calculados si es nuestro, base stats si es del rival
        if is_ally and poke.species in self.team_stats:
            stats = self.team_stats[poke.species]
        else:
            # Para rivales usamos base stats del dataset (estimación)
            poke_info = get_pokemon(poke.species, self.data["pokemon"])
            if poke_info:
                stats = {k: v for k, v in poke_info["stats"].items()}
            else:
                stats = {}

        # Modificadores de stat activos
        stat_mods = {}
        if hasattr(poke, "boosts"):
            boosts_map = {
                "atk": "attack", "def": "defense",
                "spa": "special-attack", "spd": "special-defense",
                "spe": "speed",
            }
            for abbr, full in boosts_map.items():
                stat_mods[full] = getattr(poke.boosts, abbr, 0)

        # Movimientos disponibles
        moves = []
        for move in poke.moves.values():
            move_dict = self.data["moves"].get(move.id, {})
            if move_dict:
                move_dict = dict(move_dict)
                move_dict["pp_left"] = move.current_pp
            moves.append(move_dict)

        return {
            "hp_pct":          poke.current_hp_fraction,
            "types":           types,
            "stats":           stats,
            "stat_mods":       stat_mods,
            "status":          poke.status.name.lower() if poke.status else None,
            "moves":           [m.get("name", "") for m in moves if m],
            "tera_available":  not poke.terastallized,
            "tera_type":       None,   # poke-env no siempre expone el tipo tera del rival
            "item":            poke.item if hasattr(poke, "item") else None,
        }

    def describe_embedding(self, obs: np.ndarray) -> None:
        """
        Imprime una descripción legible del vector de observación.
        Útil para debugging.
        """
        print(f"Observation vector:")
        print(f"  Shape:  {obs.shape}")
        print(f"  Min:    {obs.min():.4f}")
        print(f"  Max:    {obs.max():.4f}")
        print(f"  Mean:   {obs.mean():.4f}")
        print(f"  Non-zero: {(obs != 0).sum()} / {len(obs)}")


# ── Entry point de prueba ─────────────────────────────────────────

if __name__ == "__main__":
    print("Verificando VGCEnv...")
    team_path = Path(__file__).resolve().parent.parent / "team.txt"

    env = VGCEnv(team_path=team_path)

    print(f"  observation_space: {env.observation_space}")
    print(f"  action_space:      {env.action_space}")
    print(f"  n_actions:         {env.action_space.n}")
    print(f"  obs_size:          {env.observation_space.shape[0]}")
    print(f"  Team cargado:      {[p['name'] for p in env.team]}")
    print(f"✓ VGCEnv OK (poke-env {'disponible' if POKE_ENV_AVAILABLE else 'NO instalado'})")
