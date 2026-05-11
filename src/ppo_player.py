"""Playable PPO/RecurrentPPO policy for poke-env battles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.field import Field
from poke_env.battle.pokemon import Pokemon as PokemonObj
from poke_env.battle.weather import Weather
from poke_env.environment import DoublesEnv
from poke_env.player import Player
from poke_env.player.battle_order import DoubleBattleOrder

from src.state_encoder import StateEncoder
from src.utils import calc_all_stats, get_pokemon, load_all_data, parse_team


WEATHER_MAP = {
    Weather.RAINDANCE: "rain",
    Weather.PRIMORDIALSEA: "rain",
    Weather.SUNNYDAY: "sun",
    Weather.DESOLATELAND: "sun",
    Weather.SANDSTORM: "sandstorm",
    Weather.SNOWSCAPE: "snow",
    Weather.HAIL: "snow",
}

TERRAIN_MAP = {
    Field.GRASSY_TERRAIN: "grassy",
    Field.ELECTRIC_TERRAIN: "electric",
    Field.PSYCHIC_TERRAIN: "psychic",
    Field.MISTY_TERRAIN: "misty",
}


class PPOPolicyPlayer(Player):
    """Wrap a Stable-Baselines PPO/RecurrentPPO checkpoint as a Showdown bot."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        team_path: str | Path = "team.txt",
        device: str = "cpu",
        deterministic: bool = True,
        strict_actions: bool = False,
        record_decisions: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"No existe el checkpoint PPO: {self.checkpoint_path}")

        self.team_path = Path(team_path)
        self.device = device
        self.deterministic = deterministic
        self.strict_actions = strict_actions
        self.record_decisions = record_decisions
        self.decision_log: list[dict[str, Any]] = []
        self._recurrent_states: dict[str, Any] = {}

        self.data = load_all_data()
        self.team_list = parse_team(self.team_path)
        self._team_stats: dict[str, dict[str, Any]] = {}
        for pokemon in self.team_list:
            stats = calc_all_stats(pokemon, self.data["pokemon"])
            self._team_stats[pokemon["name"]] = stats

        self.encoder = StateEncoder(self.data["type_chart"], self.data["moves"])
        self.model, self.model_kind, self.is_recurrent = self._load_model()

    def _load_model(self):
        errors: list[str] = []

        try:
            from sb3_contrib import RecurrentPPO

            model = RecurrentPPO.load(self.checkpoint_path, device=self.device)
            return model, "RecurrentPPO", True
        except ModuleNotFoundError as exc:
            errors.append(f"RecurrentPPO no disponible: {exc}")
        except Exception as exc:
            errors.append(f"RecurrentPPO no pudo cargar el checkpoint: {exc}")

        try:
            from stable_baselines3 import PPO

            model = PPO.load(self.checkpoint_path, device=self.device)
            return model, "PPO", False
        except Exception as exc:
            errors.append(f"PPO no pudo cargar el checkpoint: {exc}")

        details = "\n  - ".join(errors)
        raise RuntimeError(
            "No se pudo cargar el checkpoint PPO. Si es recurrente, reconstruir la "
            "imagen trainer luego de instalar sb3-contrib.\n  - " + details
        )

    def _cleanup_recurrent_states(self) -> None:
        finished = [
            tag
            for tag, battle in getattr(self, "battles", {}).items()
            if getattr(battle, "finished", False)
        ]
        for tag in finished:
            self._recurrent_states.pop(tag, None)

    def _encode_poke(self, poke: PokemonObj, is_ally: bool) -> dict[str, Any]:
        types = [pokemon_type.name.lower() for pokemon_type in poke.types if pokemon_type]
        if is_ally and poke.species in self._team_stats:
            stats = self._team_stats[poke.species]
        else:
            poke_info = get_pokemon(poke.species, self.data["pokemon"])
            stats = dict(poke_info["stats"]) if poke_info else {}

        stat_mods: dict[str, int] = {}
        boosts = getattr(poke, "boosts", None)
        if boosts:
            boost_map = {
                "atk": "attack",
                "def": "defense",
                "spa": "special-attack",
                "spd": "special-defense",
                "spe": "speed",
            }
            for short_name, full_name in boost_map.items():
                value = getattr(boosts, short_name, 0)
                if value:
                    stat_mods[full_name] = int(value)

        moves_list = []
        for move in poke.moves.values():
            move_dict = self.data["moves"].get(move.id)
            if move_dict:
                move_copy = dict(move_dict)
                move_copy["pp_left"] = move.current_pp
                moves_list.append(move_copy)

        return {
            "hp_pct": poke.current_hp_fraction,
            "types": types,
            "stats": stats,
            "stat_mods": stat_mods,
            "status": poke.status.name.lower() if poke.status else None,
            "moves": [move.get("name", "") for move in moves_list],
            "tera_available": not bool(getattr(poke, "terastallized", False)),
            "tera_type": None,
            "item": poke.item if hasattr(poke, "item") else None,
        }

    def embed_battle(self, battle: DoubleBattle) -> np.ndarray:
        own_field = [
            self._encode_poke(pokemon, is_ally=True)
            for pokemon in battle.active_pokemon
            if pokemon is not None
        ]
        rival_field = [
            self._encode_poke(pokemon, is_ally=False)
            for pokemon in battle.opponent_active_pokemon
            if pokemon is not None
        ]

        active_species = {pokemon.species for pokemon in battle.active_pokemon if pokemon}
        benched = []
        for pokemon in battle.team.values():
            if pokemon.species not in active_species and not pokemon.fainted:
                benched.append(
                    {
                        "hp_pct": pokemon.current_hp_fraction,
                        "types": [
                            pokemon_type.name.lower()
                            for pokemon_type in pokemon.types
                            if pokemon_type
                        ],
                        "status": pokemon.status.name.lower() if pokemon.status else None,
                    }
                )

        weather_enum = next(iter(battle.weather), None)
        weather = WEATHER_MAP.get(weather_enum, "none") if weather_enum else "none"
        terrain = "none"
        for field_enum in battle.fields:
            terrain_name = TERRAIN_MAP.get(field_enum)
            if terrain_name:
                terrain = terrain_name
                break

        conditions = {
            "weather": weather,
            "terrain": terrain,
            "trick_room": Field.TRICK_ROOM in battle.fields,
            "turn": battle.turn,
        }
        return self.encoder.encode_manual(own_field, rival_field, benched, conditions)

    @staticmethod
    def _normalize_action(action: Any) -> np.ndarray:
        array = np.asarray(action, dtype=np.int64)
        if array.ndim == 0:
            array = np.array([int(array), 0], dtype=np.int64)
        else:
            array = array.reshape(-1).astype(np.int64)
        if array.size < 2:
            array = np.pad(array, (0, 2 - array.size), constant_values=0)
        return array[:2]

    def _predict_action(self, observation: np.ndarray, battle: DoubleBattle) -> np.ndarray:
        tag = getattr(battle, "battle_tag", "")
        if self.is_recurrent:
            state = self._recurrent_states.get(tag)
            episode_start = np.array([state is None], dtype=bool)
            action, new_state = self.model.predict(
                observation,
                state=state,
                episode_start=episode_start,
                deterministic=self.deterministic,
            )
            self._recurrent_states[tag] = new_state
        else:
            action, _ = self.model.predict(
                observation,
                deterministic=self.deterministic,
            )
        return self._normalize_action(action)

    @staticmethod
    def _single_action_distance(predicted: int, candidate: int) -> float:
        if predicted == candidate:
            return 0.0
        if predicted < 7 or candidate < 7:
            if predicted < 7 and candidate < 7:
                return float(abs(predicted - candidate))
            return 1000.0

        predicted_base = (predicted - 7) % 20
        candidate_base = (candidate - 7) % 20
        predicted_move = predicted_base // 5
        candidate_move = candidate_base // 5
        predicted_target = predicted_base % 5
        candidate_target = candidate_base % 5
        predicted_gimmick = (predicted - 7) // 20
        candidate_gimmick = (candidate - 7) // 20
        return (
            100.0 * abs(predicted_move - candidate_move)
            + 10.0 * abs(predicted_target - candidate_target)
            + 1.0 * abs(predicted_gimmick - candidate_gimmick)
        )

    @classmethod
    def _action_distance(cls, predicted: np.ndarray, candidate: np.ndarray) -> float:
        return sum(
            cls._single_action_distance(int(pred), int(cand))
            for pred, cand in zip(predicted[:2], candidate[:2])
        )

    def _project_to_legal_action(
        self,
        action: np.ndarray,
        battle: DoubleBattle,
    ) -> tuple[np.ndarray, Any, str]:
        try:
            order = DoublesEnv.action_to_order(action, battle, fake=False, strict=True)
            return action, order, ""
        except Exception as exc:
            first_error = str(exc)

        try:
            candidates = DoubleBattleOrder.join_orders(*battle.valid_orders)
        except Exception:
            candidates = []
        legal_actions: list[tuple[np.ndarray, Any]] = []
        for candidate in candidates:
            try:
                candidate_action = DoublesEnv.order_to_action(
                    candidate,
                    battle,
                    fake=False,
                    strict=True,
                )
            except Exception:
                continue
            legal_actions.append((self._normalize_action(candidate_action), candidate))

        if legal_actions:
            projected_action, projected_order = min(
                legal_actions,
                key=lambda item: self._action_distance(action, item[0]),
            )
            return projected_action, projected_order, first_error

        fallback_order = Player.choose_random_doubles_move(battle)
        try:
            fallback_action = self._normalize_action(
                DoublesEnv.order_to_action(fallback_order, battle, fake=False, strict=False)
            )
        except Exception:
            fallback_action = action
        return fallback_action, fallback_order, first_error

    def choose_move(self, battle: DoubleBattle):
        self._cleanup_recurrent_states()
        observation = self.embed_battle(battle).astype(np.float32)
        raw_action = self._predict_action(observation, battle)
        action, order, fallback_error = self._project_to_legal_action(raw_action, battle)
        if self.strict_actions and fallback_error:
            raise ValueError(fallback_error)

        if self.record_decisions:
            self.decision_log.append(
                {
                    "battle_tag": getattr(battle, "battle_tag", ""),
                    "turn": int(getattr(battle, "turn", 0) or 0),
                    "model_kind": self.model_kind,
                    "checkpoint": str(self.checkpoint_path),
                    "raw_action": [int(item) for item in raw_action.tolist()],
                    "projected_action": [int(item) for item in action.tolist()],
                    "fallback_error": fallback_error,
                    "selected_message": getattr(order, "message", None) or str(order),
                }
            )
        return order
